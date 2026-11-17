"""Outreach pipeline: Stages 1-7 of FLOWCHART.md, plus the Stage 8 review actions.

Each stage is one function that reads/writes the run state and emits a
StageEvent (started / done / failed / skipped). The UI renders the events live;
the dashboard replays the same stored events.

CLI:  python agent.py "Company" [--name NAME] [--role ROLE] [--website URL] [--notes TEXT]
      (give --name, --role or both; Stage 1 looks up the missing one)
Batch: python batch.py prospects.csv
"""

import argparse
import json
import re
import time
from concurrent.futures import TimeoutError as FutTimeout
from datetime import date, datetime, timezone
from typing import Literal, Optional
from urllib.parse import urlsplit, urlunsplit

from pydantic import BaseModel, Field

import config as C
import prompts as PR
import observability as obs
import store
from llm import embeddings, llm, llm_fallback
from search import fetch_paragraphs as _fetch_paragraphs, search as _search
from tracing import Pool, step, tag_current


def _observed_search(q):
    """search.search plus a tool_calls record: backend that answered, latency, backends skipped."""
    t0 = time.time()
    try:
        rows = _search(q)
    except Exception as e:
        obs.record_tool(tool="search", outcome="error", latency=round(time.time() - t0, 2), n_results=0,
                        detail=f"{q['bucket']}: {str(e)[:300]}")
        raise
    obs.record_tool(tool="search", backend=rows[0].get("backend") if rows else None,
                    outcome="ok" if rows else "empty", latency=round(time.time() - t0, 2), n_results=len(rows),
                    fallbacks=rows[0].get("fallbacks") if rows else None,
                    detail=(rows[0].get("backend_errors") or "")[:300] if rows else q["bucket"])
    return rows


def _observed_fetch(url):
    """search.fetch_paragraphs plus a tool_calls record, per host (which sites block or time out)."""
    t0 = time.time()
    host = (urlsplit(url or "").hostname or "?").removeprefix("www.")
    try:
        paras = _fetch_paragraphs(url)
    except Exception as e:
        obs.record_tool(tool="fetch", backend=host, outcome="error", latency=round(time.time() - t0, 2),
                        n_results=0, detail=f"{type(e).__name__}: {str(e)[:150]}")
        raise
    obs.record_tool(tool="fetch", backend=host, outcome="ok" if paras else "empty",
                    latency=round(time.time() - t0, 2), n_results=len(paras),
                    detail=f"{sum(map(len, paras))} chars")
    return paras


# Searches and page fetches as tool steps in a LangSmith trace (no-op when tracing is off).
search_tool = step("search", "tool")(_observed_search)
fetch_paragraphs = step("fetch page", "tool")(_observed_fetch)

store.init_db()

STAGE_NAMES = {1: "Input", 2: "Plan queries", 3: "Gather sources", 4: "Extract signals",
               5: "Score & pick hook", 6: "Draft", 7: "Check draft", 8: "Review"}
TERMINAL = {"drafted", "generic", "flagged", "needs_research", "failed", "approved", "discarded"}
REVIEWABLE = ("drafted", "generic", "flagged")
GENERIC_SETTING = "generic_fallback"  # Dashboard switch, stored in the settings table
NOTES_URL = "rep-notes://pasted"


class StageError(Exception):
    """A stage failed (error or timeout). The run goes to status `failed`."""


# =============================================================================
# LLM schemas
# =============================================================================
class ExtractedSignal(BaseModel):
    type: Literal[tuple(C.SIGNAL_TYPES)]
    level: Literal["person", "company"]
    summary: str
    evidence: str
    source_url: str
    date: Optional[str] = Field(None, description="YYYY-MM-DD, or null if the source shows no date")


class Extraction(BaseModel):
    signals: list[ExtractedSignal]


class LookupCandidate(BaseModel):
    value: str = Field(description="the person's full name, or the job title, as written in the source")
    source_url: str
    evidence: str
    date: Optional[str] = Field(None, description="YYYY-MM-DD, or null if the source shows no date")
    current: bool = Field(True, description="false if the source says this is a former/past holder or job")


class Lookup(BaseModel):
    candidates: list[LookupCandidate]


class PlannedQuery(BaseModel):
    query: str
    bucket: Literal["company", "person"]
    reason: str


class QueryPlan(BaseModel):
    queries: list[PlannedQuery]


class Rating(BaseModel):
    signal_id: int
    R: int = Field(description="relevance 0-3")
    reason: str


class Ratings(BaseModel):
    ratings: list[Rating]


class DraftParts(BaseModel):
    subject: str
    hook: str
    bridge: str
    value: str
    cta: str


class Claim(BaseModel):
    claim: str
    supported: bool
    support: str = Field(description="what in the signal/profile supports it, or why not")


class Grounding(BaseModel):
    claims: list[Claim]


_llm_pool = Pool(max_workers=4)


_TRANSIENT = ("503", "429", "UNAVAILABLE", "RESOURCE_EXHAUSTED", "overloaded", "high demand",
              "timed out", "Timeout", "timeout", "DEADLINE_EXCEEDED")


def call_llm(schema, system, user, timeout=C.LLM_TIMEOUT_S):
    """Structured LLM call with a hard overall deadline. Each attempt is one
    request (the SDK's own retries are off, see llm.py) with its own request
    timeout. Overload, rate-limit and timeout errors are retried with backoff;
    retries go to the fallback model, since a 503 'high demand' usually lasts
    longer than a few seconds on the same model. Raises StageError otherwise.
    Every attempt is recorded (observability.record_llm): model, outcome, latency, tokens."""
    purpose = getattr(schema, "__name__", str(schema))
    deadline = time.time() + timeout
    last = None
    for attempt, backoff in enumerate(C.LLM_RETRY_BACKOFF_S + [None]):
        remaining = deadline - time.time()
        if remaining <= 0:
            break
        model = llm if attempt == 0 else llm_fallback
        rec = {"purpose": purpose, "model": getattr(model, "model", None), "attempt": attempt,
               "prompt_chars": len(system) + len(user)}
        usage = obs.usage_callback()
        chain = model.with_structured_output(schema)
        t0 = time.time()
        fut = _llm_pool.submit(chain.invoke, [("system", system), ("human", user)],
                               **({"config": {"callbacks": [usage]}} if usage else {}))
        try:
            out = fut.result(timeout=remaining)
        except FutTimeout:
            obs.record_llm(**rec, outcome="deadline", latency=round(time.time() - t0, 2),
                           error=f"no answer within the {timeout}s deadline")
            raise StageError(f"LLM timed out after {timeout}s ({attempt + 1} attempt(s); last error: {last})")
        except Exception as e:  # network, quota, request timeout, parse errors
            last = f"{type(e).__name__}: {str(e)[:300]}"
            retry = (backoff is not None and any(t in str(e) or t in type(e).__name__ for t in _TRANSIENT)
                     and time.time() + backoff < deadline)
            obs.record_llm(**rec, outcome="retry" if retry else "error", latency=round(time.time() - t0, 2),
                           error=last)
            if retry:
                time.sleep(backoff)
                continue
            raise StageError(f"LLM error: {last}")
        tokens = {"input_tokens": getattr(usage, "input_tokens", None),
                  "output_tokens": getattr(usage, "output_tokens", None)}
        if out is None:
            obs.record_llm(**rec, outcome="empty", latency=round(time.time() - t0, 2),
                           error="no parseable structured output", **tokens)
            raise StageError("LLM returned no parseable structured output")
        obs.record_llm(**rec, outcome="ok", latency=round(time.time() - t0, 2), **tokens)
        return out
    raise StageError(f"LLM error after retries: {last}")


# =============================================================================
# Events
# =============================================================================
class Emitter:
    """Stores finished StageEvents and forwards every event to the UI callback."""

    def __init__(self, run_id, on_event=None):
        self.run_id = run_id
        self.on_event = on_event or (lambda ev: None)
        self._open = {}  # stage -> (t0, started_iso, label)

    def _notify(self, ev):
        # A rendering bug in the UI must not fail the pipeline. (Streamlit's rerun/stop
        # signals are BaseExceptions and still propagate; callers handle those.)
        try:
            self.on_event(ev)
        except Exception as e:
            print(f"[ui callback error ignored] {type(e).__name__}: {e}")

    def start(self, stage, label=None):
        label = label or STAGE_NAMES[stage]
        self._open[stage] = (time.time(), store.now_iso(), label)
        obs.bind_stage(stage)
        store.update_run(self.run_id, current_stage=stage)
        self._notify({"run_id": self.run_id, "stage": stage, "stage_name": label,
                      "status": "started", "summary": "running...", "payload": None})

    def finish(self, stage, status, summary, payload=None, label=None):
        t, started_iso, open_label = self._open.pop(stage, (time.time(), store.now_iso(), None))
        ev = {"run_id": self.run_id, "stage": stage, "stage_name": label or open_label or STAGE_NAMES[stage],
              "status": status, "started_at": started_iso, "ended_at": store.now_iso(),
              "elapsed": round(time.time() - t, 2), "summary": summary, "payload": payload}
        store.add_event(ev)
        self._notify(ev)
        return ev

    def fail_open(self, summary):
        """Close every stage still open (after an error/interrupt) as failed.
        Returns the stage numbers closed."""
        stages = list(self._open)
        for st in stages:
            try:
                self.finish(st, "failed", summary)
            except BaseException:
                pass
        return stages


# =============================================================================
# Helpers
# =============================================================================
def norm_url(u):
    u = (u or "").strip()
    if u == NOTES_URL:
        return u
    try:
        p = urlsplit(u)
    except ValueError:
        return u.lower()
    host = p.netloc.lower().removeprefix("www.")
    path = p.path.rstrip("/")
    return urlunsplit((p.scheme.lower() or "https", host, path, p.query, ""))


def domain_of(website):
    w = (website or "").strip()
    if not w:
        return ""
    if "://" not in w:
        w = "https://" + w
    return urlsplit(w).netloc.lower().removeprefix("www.")


def parse_date(s):
    """Parse an ISO-ish date string to a date, or None."""
    if not s:
        return None
    s = str(s).strip()
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")[:25]).date()
    except ValueError:
        pass
    try:
        return date.fromisoformat(s[:10])
    except ValueError:
        return None


_WORD = re.compile(r"[a-z0-9]+")
_STOP = {"the", "a", "an", "of", "to", "in", "and", "for", "on", "its", "is", "at", "by", "with",
         "as", "has", "from", "that", "it", "this", "was", "be", "new"}


def words(s):
    return [w for w in _WORD.findall((s or "").lower())]


def jaccard(a, b, exclude=frozenset()):
    sa = set(words(a)) - _STOP - exclude
    sb = set(words(b)) - _STOP - exclude
    return len(sa & sb) / len(sa | sb) if sa and sb else 0.0


def word_count(s):
    return len(re.findall(r"\b[\w'’-]+\b", s or ""))


def seller_text():
    s = C.SELLER_PROFILE
    lines = [f"Product: {s['product']} - {s['one_liner']}",
             f"Buyers: {', '.join(s['buyers'])}",
             "Problems solved:", *[f"- {p}" for p in s["problems_solved"]],
             "Proof points:", *([f"- {p}" for p in s["proof_points"]] or ["- (none; do not cite any)"])]
    return "\n".join(lines)


def assemble(prospect, parts):
    first = prospect["name"].split()[0]
    body = (f"Hi {first},\n\n{parts['hook'].strip()} {parts['bridge'].strip()}\n\n"
            f"{parts['value'].strip()}\n\n{parts['cta'].strip()}\n\n{C.SELLER_PROFILE['sign_off']}")
    return body


def core_text(parts):
    return " ".join(parts[k].strip() for k in ("hook", "bridge", "value", "cta"))


# =============================================================================
# Stage 1 - Input, plus lookup of a missing name or role (search tool + LLM)
# =============================================================================
def validate_input(p):
    """Returns the missing required fields (empty = valid). Company is required,
    plus the prospect name or role (or both); Stage 1 looks up a missing one."""
    missing = [] if (p.get("company") or "").strip() else ["company"]
    if not (p.get("name") or "").strip() and not (p.get("role") or "").strip():
        missing.append("name or role")
    return missing


_HONORIFIC = re.compile(r"^(?:(?:dr|mr|mrs|ms|mx|prof|sir|dame)\b\.?\s*)+", re.I)


def clean_name(n):
    """'Dr. Jane Doe, CPA' -> 'Jane Doe'. The draft greets by the first word."""
    n = " ".join((n or "").split()).split(",")[0]
    return _HONORIFIC.sub("", n).strip(" .")


def lookup_queries(p, field):
    """field='name': who is the {role} of {company}; field='role': {name}'s title there."""
    fill = {"name": p["name"], "role": p["role"], "company": p["company"]}
    if field == "name":
        templates, site = C.NAME_LOOKUP_QUERIES, C.NAME_LOOKUP_SITE_QUERY
    else:
        templates, site = C.ROLE_LOOKUP_QUERIES, C.ROLE_LOOKUP_SITE_QUERY
    # "person" bucket: web results first (LinkedIn and team pages carry names and titles).
    qs = [{"bucket": "person", "query": t.format(**fill)} for t in templates]
    dom = domain_of(p.get("website"))
    if dom:
        qs.append({"bucket": "person", "query": site.format(domain=dom, **fill)})
    return qs


def _on_site(url, website):
    dom = domain_of(website)
    host = urlsplit(url or "").netloc.lower().removeprefix("www.")
    return bool(dom) and (host == dom or host.endswith("." + dom))


def _title_words(title):
    return [w for w in words(title) if w not in _STOP]


def _title_in(title, text):
    """The title is written in the source, word for word or as its acronym
    ('Chief Financial Officer' ~ 'CFO')."""
    tw, sw = _title_words(title), set(words(text))
    return bool(tw) and (set(tw) <= sw or (len(tw) > 1 and "".join(w[0] for w in tw) in sw))


def _role_key(title):
    """Groups 'Chief Financial Officer' with 'CFO' so they don't look like rivals."""
    tw = _title_words(title)
    return "".join(w[0] for w in tw) if len(tw) > 1 else " ".join(tw)


def _check_candidate(c, field, p, src):
    """Code checks on one LLM lookup candidate. Returns a drop reason, or None."""
    if not c["current"]:
        return "source says this is a former holder / past job"
    text = f"{src.get('title') or ''} {src.get('snippet') or ''}"
    if field == "name":
        if len(c["value"].split()) < 2:
            return "not a full name"
        if not _mentions_person(text, c["value"]):
            return "name is not written in the source"
    else:
        if not _mentions_person(text, p["name"]):
            return "source does not mention the prospect"
        if not _title_in(c["value"], text):
            return "title is not written in the source"
    if (src["origin"] != "rep_provided" and not mentions_company(text, p["company"])
            and not _on_site(src["url"], p.get("website"))):
        return "source does not mention the company (possible namesake or other company)"
    return None


@step('lookup prospect')
def lookup_prospect(p, field):
    """Find the missing name (field='name') or role (field='role') by web search.
    Returns (value or None, info); info lists every candidate and drop reason.
    Raises StageError if every search fails (and no notes) or the LLM call fails."""
    queries = lookup_queries(p, field)
    sources, counts, errors = gather_sources(queries, p["notes"], rank=False)  # ~20 results: nothing to cut
    info = {"field": field, "queries": queries, **counts, "errors": errors, "sources": sources,
            "candidates": [], "dropped": [], "chosen": None}
    if not sources:
        return None, info

    fmt = {"name": p["name"], "role": p["role"], "company": p["company"]}
    sys = (PR.LOOKUP_NAME_SYSTEM if field == "name" else PR.LOOKUP_ROLE_SYSTEM).format(**fmt)
    user = PR.LOOKUP_USER.format(today=date.today().isoformat(), sources=_sources_block(sources))
    raw = call_llm(Lookup, sys, user).candidates

    by_url = {norm_url(s["url"]): s for s in sources}
    groups = {}
    for c in (x.model_dump() for x in raw):
        c["value"] = clean_name(c["value"]) if field == "name" else " ".join(c["value"].split()).strip(" ,.")
        src = by_url.get(norm_url(c["source_url"]))
        reason = "source_url was not gathered" if src is None else _check_candidate(c, field, p, src)
        if reason:
            info["dropped"].append({**c, "drop_reason": reason})
            continue
        d = parse_date(c["date"]) or parse_date(src.get("published_date"))
        key = " ".join(words(c["value"])) if field == "name" else _role_key(c["value"])
        g = groups.setdefault(key, {"value": c["value"], "sources": [], "evidence": [], "latest": None})
        if src["url"] not in g["sources"]:
            g["sources"].append(src["url"])
            g["evidence"].append(c["evidence"])
        if d and (g["latest"] is None or d.isoformat() > g["latest"]):
            g["latest"] = d.isoformat()
    # Most supporting sources first; ties -> most recent dated source.
    info["candidates"] = sorted(groups.values(), key=lambda g: (len(g["sources"]), g["latest"] or ""),
                                reverse=True)
    if info["candidates"]:
        info["chosen"] = info["candidates"][0]["value"]
    return info["chosen"], info


def lookup_record(info):
    """The part of a lookup the review card shows (stored on the run)."""
    best = info["candidates"][0] if info["candidates"] else None
    return {"field": info["field"], "value": info["chosen"],
            "evidence": best["evidence"][0] if best else None,
            "sources": best["sources"] if best else [],
            "latest": best["latest"] if best else None,
            "alternatives": [{"value": c["value"], "sources": c["sources"]} for c in info["candidates"][1:]]}


# =============================================================================
# Stage 2 - Plan queries (code)
# =============================================================================
def plan_queries(p):
    fill = {"name": p["name"].strip(), "company": p["company"].strip(),
            "function": C.SELLER_PROFILE["function"]}
    qs = [{"bucket": "company", "query": t.format(**fill)} for t in C.COMPANY_QUERIES]
    qs += [{"bucket": "person", "query": t.format(**fill)} for t in C.PERSON_QUERIES]
    dom = domain_of(p.get("website"))
    if dom:
        qs.append({"bucket": "company", "query": C.WEBSITE_QUERY.format(domain=dom)})
    return qs


# =============================================================================
# Stage 3 - Gather sources (search tool)
# =============================================================================
@step('gather sources')
def gather_sources(queries, notes, rank=True, exclude=frozenset()):
    """Run the queries, dedupe the results and (rank=True) keep the top
    MAX_SOURCES by similarity to the seller (see rank_sources). Pasted notes are
    always added. `exclude` = normalised URLs seen in earlier research rounds.
    Returns (kept sources, info, errors); info["cut"] holds the results that
    were scored but not kept."""
    per_query, errors = {}, {}
    pool = Pool(max_workers=C.SEARCH_WORKERS)
    try:
        futs = {q["query"]: pool.submit(search_tool, q) for q in queries}
        # Each query may walk up to 4 backends; bound the whole stage.
        deadline = time.time() + C.SEARCH_TIMEOUT_S * 4
        for qtext, fut in futs.items():
            try:
                per_query[qtext] = fut.result(timeout=max(1, deadline - time.time()))
            except FutTimeout:
                errors[qtext] = "timeout"
                per_query[qtext] = []
                obs.record_tool(tool="search", outcome="timeout", latency=C.SEARCH_TIMEOUT_S * 4, n_results=0,
                                detail=f"stage gave up waiting: {qtext[:100]}")
            except Exception as e:
                errors[qtext] = f"{type(e).__name__}: {e}"[:600]
                per_query[qtext] = []
    finally:
        pool.shutdown(wait=False, cancel_futures=True)  # don't block on timed-out searches

    if len(errors) == len(queries) and not (notes or "").strip():
        raise StageError(f"all {len(queries)} searches failed: {list(errors.values())[0]}")

    counts = {q: len(v) for q, v in per_query.items()}
    # Dedupe URLs, taking results round-robin across queries (so equal scores keep each query's best first).
    bucket = {q["query"]: q["bucket"] for q in queries}
    seen, pool = set(), []
    lists = [list(v) for v in per_query.values()]
    while any(lists):
        for lst in lists:
            if lst:
                s = lst.pop(0)
                key = norm_url(s["url"])
                if key not in seen and key not in exclude:
                    seen.add(key)
                    pool.append({**s, "bucket": bucket.get(s["query"])})
    if rank:
        sources, cut, method = rank_sources(pool)
    else:
        sources, cut, method = pool[:C.MAX_SOURCES], pool[C.MAX_SOURCES:], None
    if (notes or "").strip():
        sources.append({"url": NOTES_URL, "title": "Notes pasted by rep", "snippet": notes.strip(),
                        "published_date": None, "query": None, "origin": "rep_provided", "kept": 1})
    backends = {}
    for s in sources:
        b = s.get("backend") or s["origin"]
        backends[b] = backends.get(b, 0) + 1
    return sources, {"per_query": counts, "backends": backends, "gathered": len(pool),
                     "ranking": method, "cut": cut}, errors


def source_anchors():
    return list(C.SELLER_PROFILE["problems_solved"]) + list(C.TRIGGER_ANCHORS)


def _source_text(s):
    return f"{s.get('title') or ''}. {s.get('snippet') or ''}"[:C.SNIPPET_CHARS]


def _unit(v):
    n = sum(x * x for x in v) ** 0.5 or 1.0
    return [x / n for x in v]


_anchor_vecs = {}


def _embedding_scores(anchors, texts):
    """Best cosine similarity of each text to any anchor -> [(score, anchor index)]."""
    def run():
        key = tuple(anchors)
        if key not in _anchor_vecs:  # anchors are fixed per process: embed once
            _anchor_vecs[key] = [_unit(v) for v in embeddings.embed_documents(list(anchors))]
        return _anchor_vecs[key], [_unit(v) for v in embeddings.embed_documents(texts)]
    t0 = time.time()
    try:
        A, T = _llm_pool.submit(run).result(timeout=C.LLM_TIMEOUT_S)
    except BaseException as e:
        obs.record_tool(tool="embedding", backend="gemini-embedding-001", outcome="error",
                        latency=round(time.time() - t0, 2), n_results=len(texts),
                        detail=f"{type(e).__name__}: {str(e)[:150]}")
        raise
    obs.record_tool(tool="embedding", backend="gemini-embedding-001", outcome="ok",
                    latency=round(time.time() - t0, 2), n_results=len(texts))
    return [max((sum(x * y for x, y in zip(t, a)), j) for j, a in enumerate(A)) for t in T]


def _stem(w):
    for suf in ("ing", "ed", "es", "s"):
        if w.endswith(suf) and len(w) - len(suf) >= 4:
            return w[:-len(suf)]
    return w


def _bag(s):
    return {_stem(w) for w in words(s) if w not in _STOP and len(w) > 2}


def _keyword_scores(anchors, texts):
    """Fallback: share of an anchor's words found in the text, best anchor."""
    A = [_bag(a) for a in anchors]
    out = []
    for t in texts:
        tb = _bag(t)
        out.append(max((len(a & tb) / len(a) if a else 0.0, j) for j, a in enumerate(A)))
    return out


@step('rank sources')
def rank_sources(pool):
    """Score every result against the anchors (seller problems + trigger events)
    and keep the top MAX_SOURCES, always including the best MIN_PERSON_SOURCES
    person-query results. Returns (kept, cut, method); each result gets sim, anchor, kept."""
    if not pool:
        return [], [], None
    anchors, texts = source_anchors(), [_source_text(s) for s in pool]
    try:
        scores, method = _embedding_scores(anchors, texts), "embedding"
    except Exception as e:  # quota, network, timeout: rank anyway, never fail the run here
        scores = _keyword_scores(anchors, texts)
        method = f"keyword overlap (embedding failed: {type(e).__name__}: {str(e)[:150]})"
    for s, (sc, j) in zip(pool, scores):
        s["sim"], s["anchor"] = round(sc, 3), anchors[j]
    order = sorted(range(len(pool)), key=lambda i: pool[i]["sim"], reverse=True)  # stable: ties keep round-robin order
    person = [i for i in order if pool[i]["bucket"] == "person"][:C.MIN_PERSON_SOURCES]
    keep = set((person + [i for i in order if i not in person])[:C.MAX_SOURCES])
    kept = [pool[i] for i in order if i in keep]
    cut = [pool[i] for i in order if i not in keep]
    for s in kept:
        s["kept"] = 1
    for s in cut:
        s["kept"] = 0
    return kept, cut, method


def _fetchable(url):
    try:
        host = (urlsplit(url or "").hostname or "").lower().removeprefix("www.")
    except ValueError:  # malformed URL from a search backend
        return False
    return bool(host) and url.startswith(("http://", "https://")) and not any(
        host == h or host.endswith("." + h) for h in C.FETCH_SKIP_HOSTS)


def page_excerpt(paras, p, limit=None):
    """The lede (first 2 paragraphs) plus paragraphs that mention the company or
    the prospect, in page order; padded with the rest if that is short."""
    limit = limit or C.FULLTEXT_CHARS
    last = (p.get("name") or "").split()[-1:]

    def relevant(t):
        return mentions_company(t, p["company"]) or bool(last and re.search(rf"\b{re.escape(last[0])}\b", t))
    picked = [i for i, t in enumerate(paras) if i < 2 or relevant(t)]
    if sum(len(paras[i]) for i in picked) < limit // 2:
        picked = list(range(len(paras)))
    out, n = [], 0
    for i in picked:
        if n >= limit:
            break
        out.append(paras[i])
        n += len(paras[i]) + 1
    return " ".join(out)[:limit]


@step('fetch full text')
def fetch_full_text(sources, p):
    """Fetch the pages of the top FETCH_TOP_N fetchable sources (already in rank
    order) and set s["fulltext"]. Failures leave the snippet in place.
    Returns {url: "ok" | error}."""
    targets = [s for s in sources if s["origin"] == "search" and _fetchable(s["url"])][:C.FETCH_TOP_N]
    if not targets:
        return {}
    status = {}
    pool = Pool(max_workers=len(targets))
    try:
        futs = {s["url"]: (s, pool.submit(fetch_paragraphs, s["url"])) for s in targets}
        deadline = time.time() + C.FETCH_TIMEOUT_S * 2
        for url, (s, fut) in futs.items():
            try:
                paras = fut.result(timeout=max(1, deadline - time.time()))
                if not any(mentions_company(t, p["company"]) for t in paras):
                    status[url] = "page does not mention the company"
                    continue
                text = page_excerpt(paras, p)
            except FutTimeout:
                status[url] = "timeout"
                continue
            except Exception as e:
                status[url] = f"{type(e).__name__}: {str(e)[:120]}"
                continue
            # Too little text means a paywall, consent page or JS-only site: keep the snippet.
            if len(text) < len(s.get("snippet") or "") + 200:
                status[url] = "no article text"
                continue
            s["fulltext"] = text
            status[url] = "ok"
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
    return status


# =============================================================================
# Stage 4 - Extract signals (LLM) + code checks + merge
# =============================================================================
def _source_body(s):
    """What Stage 4 reads: pasted notes in full, a fetched page's excerpt, else the snippet."""
    if s["origin"] == "rep_provided":
        return s.get("snippet") or ""
    if s.get("fulltext"):
        return s["fulltext"][:C.FULLTEXT_CHARS]
    return (s.get("snippet") or "")[:C.SNIPPET_CHARS]


def _sources_block(sources):
    blocks = []
    for i, s in enumerate(sources, 1):
        kind = "full article excerpt" if s.get("fulltext") and s["origin"] != "rep_provided" else "text"
        blocks.append(f"[S{i}] url: {s['url']}\ntitle: {s.get('title') or ''}\n"
                      f"published: {s.get('published_date') or 'unknown'}\n{kind}: {_source_body(s)}")
    return "\n\n".join(blocks)


_LEGAL = r"(?:inc|incorporated|ltd|limited|llc|llp|plc|corp|corporation|co|company|gmbh|ag|sa|pvt|private|pte)"


def company_core(company):
    """'Stripe, Inc.' -> 'Stripe'; 'Acme Pvt. Ltd.' -> 'Acme'."""
    core = (company or "").strip()
    while True:
        new = re.sub(rf"[\s,]+{_LEGAL}\.?$", "", core, flags=re.I).strip(" ,.")
        if new == core or not new:
            return core
        core = new


def mentions_company(text, company):
    core = company_core(company)
    return bool(core) and re.search(rf"(?<!\w){re.escape(core)}(?!\w)", text or "", re.I) is not None


def _has_word(text, phrase):
    return re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", text) is not None


def _mentions_person(text, name, full_name_only=False):
    """Whole-word match ('Doe' must not match 'does'). Short texts may give first
    and last name separately; long ones (full articles) need the full name."""
    t = " ".join((text or "").lower().split())
    n = " ".join((name or "").lower().split())
    if not n:
        return False
    if _has_word(t, n):
        return True
    parts = [x for x in n.split() if len(x) > 1]
    return not full_name_only and len(parts) >= 2 and _has_word(t, parts[0]) and _has_word(t, parts[-1])


@step('extract signals')
def extract_signals(p, sources, existing=()):
    """Returns (new signals, dropped, grown). `existing` = verified signals from
    earlier research rounds: a new report of the same event is merged into it
    (it is then in `grown`) instead of becoming a new signal."""
    if not sources:
        return [], [], []
    sys = PR.EXTRACT_SYSTEM.format(types=", ".join(C.SIGNAL_TYPES), company=p["company"], name=p["name"])
    user = PR.EXTRACT_USER.format(name=p["name"], role=p["role"], company=p["company"],
                                  today=date.today().isoformat(), sources=_sources_block(sources))
    raw = call_llm(Extraction, sys, user).signals

    by_url = {norm_url(s["url"]): s for s in sources}
    kept, dropped = [], []
    for s in raw:
        sig = s.model_dump()
        src = by_url.get(norm_url(sig["source_url"]))
        if src is None:
            dropped.append({**sig, "drop_reason": "source_url was not gathered in Stage 3"})
            continue
        sig["source_url"] = src["url"]  # canonical form
        if parse_date(sig["date"]) is None:
            sig["date"] = None
        if sig["date"] is None and parse_date(src.get("published_date")):
            sig["date"] = parse_date(src["published_date"]).isoformat()
        short = f"{src.get('title') or ''} {src.get('snippet') or ''}"
        full = src.get("fulltext") or ""
        # In a long article, first and last name apart are weak evidence: require the full name there.
        person_ok = _mentions_person(short, p["name"]) or _mentions_person(full, p["name"], full_name_only=True)
        if (sig["level"] == "person" and src["origin"] != "rep_provided"
                and not (person_ok and mentions_company(f"{short} {full}", p["company"]))):
            dropped.append({**sig, "drop_reason": "person-level, but source does not mention both "
                                                  "the prospect's name and company (possible namesake)"})
            continue
        kept.append(sig)
    exclude = {w.lower() for w in re.findall(r"\w+", f"{p['name']} {p['company']}")}
    new, grown = merge_signals(kept, exclude, existing)
    return new, dropped, grown


_GENERIC_CAPS = {"the", "a", "an", "ai", "crm", "ceo", "cto", "cfo", "coo", "vp", "inc", "ltd", "llc", "co",
                 "january", "february", "march", "april", "may", "june", "july", "august", "september",
                 "october", "november", "december", "series", "new", "its", "his", "her", "their"}


def key_entities(text, exclude):
    """Distinctive capitalised tokens (e.g. 'Warmly', 'Q3') minus the prospect/company names."""
    toks = re.findall(r"\b(?:[A-Z][a-zA-Z0-9]+|[A-Z]\d+)\b", text or "")
    return {t.lower() for t in toks} - _GENERIC_CAPS - exclude


def merge_signals(sigs, exclude=frozenset(), existing=()):
    """Merge the same event reported by several outlets (heuristic): same type and
    level, dates within MERGE_MAX_DAYS (or both undated), and either overlapping
    summaries (Jaccard) or a shared distinctive entity such as the acquired company.
    `existing` signals (earlier research rounds) can absorb new reports; they are
    updated in place. Returns (new merged signals, existing signals that grew)."""
    groups = []
    for e in existing:
        d0 = parse_date(e["date"])
        groups.append({"sig": e, "dates": [d0] if d0 else [], "ents": key_entities(e["summary"], exclude),
                       "old": True, "grew": False})
    for s in sigs:
        d = parse_date(s["date"])
        s_ents = key_entities(s["summary"], exclude)
        for g in groups:
            m, mdates = g["sig"], g["dates"]
            # An undated report is compatible with any date (outlets often omit it).
            same_time = d is None or not mdates or any(abs((d - x).days) <= C.MERGE_MAX_DAYS for x in mdates)
            shared = s_ents & g["ents"]
            # A shared name identifies the event only for counterparty-type events (the acquired
            # company, the partner, the product); elsewhere require a shared period token like 'Q3'.
            entity_match = bool(shared) and (s["type"] in C.MERGE_ENTITY_TYPES
                                             or any(ch.isdigit() for t in shared for ch in t))
            similar = jaccard(m["summary"], s["summary"], exclude) >= C.MERGE_MIN_JACCARD or entity_match
            if m["type"] == s["type"] and m["level"] == s["level"] and same_time and similar:
                if s["source_url"] not in m["source_urls"]:
                    m["source_urls"].append(s["source_url"])
                    g["grew"] = True
                if d:
                    mdates.append(d)
                    new_date = min(mdates).isoformat()  # earliest report = event date
                    g["grew"] |= new_date != m["date"]
                    m["date"] = new_date
                g["ents"] |= s_ents
                break
        else:
            groups.append({"sig": {k: v for k, v in s.items() if k != "source_url"} | {"source_urls": [s["source_url"]]},
                           "dates": [d] if d else [], "ents": set(s_ents), "old": False, "grew": False})
    return ([g["sig"] for g in groups if not g["old"]],
            [g["sig"] for g in groups if g["old"] and g["grew"]])


# =============================================================================
# Stage 5 - Score & pick (LLM relevance, code for the rest)
# =============================================================================
def recency_T(d, today=None):
    d = parse_date(d)
    if d is None:
        return C.UNDATED_T
    days = ((today or date.today()) - d).days
    if days < -1:  # implausible future date: treat as undated
        return C.UNDATED_T
    for max_days, t in C.RECENCY_BANDS:
        if days <= max_days:
            return t
    return C.UNDATED_T


def compute_score(s):
    """Code part of the score (needs s["R"]). Rerun when a signal's date changes."""
    s["T"] = recency_T(s["date"])
    s["P"] = C.P_PERSON if s["level"] == "person" else C.P_COMPANY
    s["score"] = round((s["R"] / 3) * (C.W_RECENCY * s["T"] + C.W_PERSONAL * s["P"]), 3)
    s["passed_gate"] = int(s["R"] >= C.GATE_MIN_R)


@step('rate signals')
def rate_signals(p, signals):
    """LLM relevance R for these signals (one call), then the code score. In place."""
    if not signals:
        return
    listing = "\n".join(f"[{i}] ({s['type']}, {s['level']}, date {s['date'] or 'unknown'}) "
                        f"{s['summary']} | evidence: \"{s['evidence']}\"" for i, s in enumerate(signals))
    sys = PR.RELEVANCE_SYSTEM.format(seller=seller_text(), name=p["name"], role=p["role"], company=p["company"])
    ratings = {r.signal_id: r for r in call_llm(Ratings, sys, PR.RELEVANCE_USER.format(signals=listing)).ratings}
    for i, s in enumerate(signals):
        r = ratings.get(i)
        s["R"] = max(0, min(3, int(r.R))) if r else 0
        s["R_reason"] = r.reason if r else "not rated by the LLM (treated as 0)"
        compute_score(s)


def rank_signals(signals):
    """Highest score first; ties -> more recent, then person-level. Sets s["rank"]."""
    def key(s):
        d = parse_date(s["date"])
        return (s["score"], d.toordinal() if d else 0, s["P"])
    ranked = sorted(signals, key=key, reverse=True)
    for rank, s in enumerate(ranked, 1):
        s["rank"] = rank
    return ranked


# =============================================================================
# Research rounds 2+ (when no signal passes the gate)
# =============================================================================
def research_gaps(signals, n_kept):
    """Why round so far found no hook. 'no_relevant' always applies when we retry."""
    gaps = []
    if n_kept < C.FEW_RESULTS:
        gaps.append("few_results")
    if not any(s["level"] == "person" for s in signals):
        gaps.append("no_person")
    gaps.append("no_relevant")
    return gaps


def _query_norm(q):
    return " ".join(q.lower().replace('"', " ").split())


def gap_queries(p, gaps, already):
    """Round 2: fixed templates for the gaps, round-robin across gaps, capped.
    Templates needing a website are skipped when none was given."""
    dom = domain_of(p.get("website"))
    fill = {"name": p["name"], "role": p["role"], "company": p["company"], "domain": dom}
    per_gap = []
    for g in gaps:
        qs = []
        for t in C.GAP_QUERIES.get(g, []):
            if "{domain}" in t and not dom:
                continue
            q = t.format(**fill)
            if _query_norm(q) not in already:
                qs.append({"bucket": "person" if g == "no_person" else "company", "query": q, "gap": g})
        per_gap.append(qs)
    out = []
    while any(per_gap) and len(out) < C.MAX_GAP_QUERIES:
        for qs in per_gap:
            if qs and len(out) < C.MAX_GAP_QUERIES:
                q = qs.pop(0)
                if _query_norm(q["query"]) not in {_query_norm(x["query"]) for x in out}:
                    out.append(q)
    return out


def check_planned_query(q, p, previous):
    """Code guardrails on one LLM-written query. Returns a reject reason, or None."""
    text = q.strip()
    if not text:
        return "empty"
    if len(text) > C.LLM_QUERY_MAX_CHARS:
        return f"longer than {C.LLM_QUERY_MAX_CHARS} characters"
    if not mentions_company(text, p["company"]):
        return "does not contain the company name"
    dom = domain_of(p.get("website"))
    for site in re.findall(r"site:(\S+)", text, re.I):
        host = domain_of(site.strip('"()'))
        if not dom or not (host == dom or host.endswith("." + dom)):
            return f"site: is only allowed for the company's own website ({site})"
    # Ignore the names (in every query) and search operators when comparing.
    names = {w.lower() for w in re.findall(r"\w+", f"{p['name']} {p['company']}")} | {"or", "and", "site"}
    for prev in previous:
        same_words = set(words(text)) - {"or", "and"} == set(words(prev)) - {"or", "and"}
        if same_words or jaccard(text, prev, names) >= C.QUERY_DUP_JACCARD:
            return f"too similar to an earlier query: {prev}"
    return None


@step('LLM queries')
def llm_queries(p, all_queries, signals, gaps):
    """Round 3: the LLM writes new queries from what was found and what is missing;
    code keeps at most LLM_QUERY_MAX that pass check_planned_query.
    Returns (accepted, rejected)."""
    findings = "\n".join(f"- ({s['type']}, {s['level']}, R={s['R']}) {s['summary']} | why: {s['R_reason']}"
                         for s in signals) or "- nothing usable found"
    website = f" (website: {p['website']})" if p.get("website") else ""
    sys = PR.QUERY_PLAN_SYSTEM.format(seller=seller_text(), name=p["name"], role=p["role"], company=p["company"],
                                      website=website, max_n=C.LLM_QUERY_MAX, max_chars=C.LLM_QUERY_MAX_CHARS)
    user = PR.QUERY_PLAN_USER.format(queries="\n".join(f"- {q}" for q in all_queries), findings=findings,
                                     gaps=", ".join(gaps))
    plan = call_llm(QueryPlan, sys, user).queries
    accepted, rejected, previous = [], [], list(all_queries)
    for q in plan:
        reason = check_planned_query(q.query, p, previous)
        if reason is None and len(accepted) >= C.LLM_QUERY_MAX:
            reason = f"over the limit of {C.LLM_QUERY_MAX} queries"
        if reason:
            rejected.append({"query": q.query, "reason": q.reason, "rejected": reason})
            continue
        accepted.append({"bucket": q.bucket, "query": " ".join(q.query.split()), "reason": q.reason})
        previous.append(q.query)  # also blocks near-duplicates within the plan
    return accepted, rejected


# =============================================================================
# Stage 6 - Draft (LLM)
# =============================================================================
def generic_fallback_on():
    """The Dashboard switch: write a generic draft instead of stopping at needs_research."""
    return bool(store.get_setting(GENERIC_SETTING, C.GENERIC_FALLBACK_DEFAULT))


def draft_email(p, sig, failures=None, previous=None):
    """sig=None writes the generic (role-based, no-signal) draft."""
    s = C.SELLER_PROFILE
    feedback = ""
    if failures:
        feedback = PR.DRAFT_FEEDBACK.format(failures="\n".join(f"- {f}" for f in failures),
                                            previous=json.dumps(previous, indent=2))
    if sig is None:
        sys = PR.DRAFT_GENERIC_SYSTEM.format(seller=seller_text(), company=p["company"], tone=s["tone"],
                                             subject_max=C.SUBJECT_MAX_WORDS, ctas=" | ".join(s["allowed_ctas"]),
                                             min_words=C.BODY_MIN_WORDS, max_words=C.BODY_MAX_WORDS)
        user = f"PROSPECT: {p['name']}, {p['role'] or 'role unknown'} at {p['company']}\n{feedback}"
        parts = call_llm(DraftParts, sys, user).model_dump()
        return parts, assemble(p, parts)
    sys = PR.DRAFT_SYSTEM.format(seller=seller_text(), company=p["company"], tone=s["tone"],
                                 subject_max=C.SUBJECT_MAX_WORDS, ctas=" | ".join(s["allowed_ctas"]),
                                 min_words=C.BODY_MIN_WORDS, max_words=C.BODY_MAX_WORDS,
                                 signal_type=sig["type"], framing=C.HOOK_FRAMING.get(sig["type"], C.HOOK_FRAMING["other"]))
    user = PR.DRAFT_USER.format(name=p["name"], role=p["role"], company=p["company"], signal_type=sig["type"],
                                summary=sig["summary"], evidence=sig["evidence"], date=sig["date"] or "unknown",
                                feedback=feedback)
    parts = call_llm(DraftParts, sys, user).model_dump()
    return parts, assemble(p, parts)


# =============================================================================
# Stage 7 - Check (code + one LLM grounding check)
# =============================================================================
def _norm_phrase(s):
    return " ".join(words(s))


@step('check draft')
def check_draft(p, sig, parts, body):
    checks = []
    core = core_text(parts)
    all_text = f"{parts['subject']} {core}"

    n_sub, n_body = word_count(parts["subject"]), word_count(core)
    checks.append({"check": "length",
                   "passed": n_sub <= C.SUBJECT_MAX_WORDS and C.BODY_MIN_WORDS <= n_body <= C.BODY_MAX_WORDS,
                   "detail": f"subject {n_sub} words (max {C.SUBJECT_MAX_WORDS}); body {n_body} words "
                             f"({C.BODY_MIN_WORDS}-{C.BODY_MAX_WORDS})"})

    low = all_text.lower()
    hits = [b for b in C.BANNED_PHRASES if b in low]
    checks.append({"check": "banned_phrases", "passed": not hits,
                   "detail": f"found: {', '.join(hits)}" if hits else "none found"})

    name_problems = []
    core_co = company_core(p["company"])
    if p["company"] not in all_text and not re.search(rf"(?<!\w){re.escape(core_co)}(?!\w)", all_text):
        name_problems.append(f"company not written exactly as '{p['company']}'")
    if re.match(r"\s*(hi|hello|dear|hey)\b", parts["hook"], re.I):
        name_problems.append("hook starts with its own greeting")
    first, last = p["name"].split()[0], p["name"].split()[-1]
    prose = " ".join(parts[k] for k in ("hook", "bridge", "value"))  # the CTA is fixed text
    # Mid-sentence capitalised words only (sentence-initial words like 'Open' are ordinary words).
    for tok in set(re.findall(r"(?<![.!?]\s)(?<!^)\b[A-Z][a-z]+\b", prose.strip())):
        # One letter off the prospect's first/last name is a likely misspelling.
        for real in {first, last}:
            if tok != real and len(real) > 3 and _edit1(tok.lower(), real.lower()):
                name_problems.append(f"'{tok}' looks like a misspelling of '{real}'")
    checks.append({"check": "names", "passed": not name_problems,
                   "detail": "; ".join(name_problems) or "company and prospect names match input"})

    allowed = {_norm_phrase(c) for c in C.SELLER_PROFILE["allowed_ctas"]}
    ask_problems = []
    if _norm_phrase(parts["cta"]) not in allowed:
        ask_problems.append(f"CTA is not one of the allowed CTAs: '{parts['cta']}'")
    pre = " ".join(parts[k] for k in ("hook", "bridge", "value"))
    if "?" in pre:
        ask_problems.append("a question appears outside the CTA (more than one ask)")
    if any(a and a in _norm_phrase(pre) for a in allowed):
        ask_problems.append("an allowed CTA also appears outside the CTA line")
    checks.append({"check": "one_ask", "passed": not ask_problems,
                   "detail": "; ".join(ask_problems) or "exactly one allowed CTA"})

    try:
        g = call_llm(Grounding, PR.GROUNDING_SYSTEM, PR.GROUNDING_USER.format(
            summary=sig["summary"] if sig else PR.GENERIC_SIGNAL_SUMMARY,
            evidence=sig["evidence"] if sig else "", date=(sig and sig["date"]) or "unknown",
            name=p["name"], role=p["role"], company=p["company"], seller=seller_text(),
            subject=parts["subject"], body=body))
        unsupported = [c.claim for c in g.claims if not c.supported]
        checks.append({"check": "grounding", "passed": not unsupported,
                       "detail": ("unsupported: " + " | ".join(unsupported)) if unsupported
                       else f"{len(g.claims)} claims, all supported",
                       "claims": [c.model_dump() for c in g.claims]})
    except StageError as e:
        # passed=None: the check could not run (infrastructure), which is not a failed check.
        checks.append({"check": "grounding", "passed": None, "detail": f"grounding check could not run: {e}"})
    return checks


def _edit1(a, b):
    """True if a and b differ by exactly one edit (insert, delete or substitute)."""
    if a == b or abs(len(a) - len(b)) > 1:
        return False
    if len(a) == len(b):
        return sum(x != y for x, y in zip(a, b)) == 1
    if len(a) > len(b):
        a, b = b, a
    i = 0
    while i < len(a) and a[i] == b[i]:
        i += 1
    return a[i:] == b[i + 1:]


# =============================================================================
# Stages 6 + 7 loop (shared by the pipeline and Regenerate)
# =============================================================================
@step('draft and check')
def draft_and_check(run_id, p, sig, em, version, replaces=None):
    """Draft, check, redraft once on failure. Returns (draft_id, status).

    `replaces` is the draft id this generation replaces (Regenerate); it is
    marked superseded only once a new draft exists, so a failed regenerate
    leaves the old draft reviewable.
    Raises StageError only if the FIRST draft cannot be written. If a redraft
    errors, the previous (failed-check) draft is kept and flagged.
    sig=None writes a generic draft (no signal); the draft is stored with kind 'generic'."""
    failures, previous, draft_id = None, None, None
    kind = "generic" if sig is None else "personalised"
    tag = "generic, " if sig is None else ""
    for attempt in range(1, C.MAX_DRAFT_ATTEMPTS + 1):
        label6 = f"Draft ({tag}v{version}, attempt {attempt})"
        em.start(6, label6)
        try:
            parts, body = draft_email(p, sig, failures, previous)
        except StageError as e:
            em.finish(6, "failed", str(e), label=label6)
            if draft_id is None:
                raise
            store.update_draft(draft_id, status="flagged")  # keep the attempt-1 draft for review
            return draft_id, "flagged"
        prior = draft_id or replaces
        draft_id = store.add_draft({"run_id": run_id, "signal_id": sig["id"] if sig else None,
                                    "version": version, "attempt": attempt, **parts, "body": body,
                                    "status": "checking", "kind": kind})
        if prior:
            store.update_draft(prior, status="superseded")
        em.finish(6, "done", f"\"{parts['subject']}\" - {word_count(core_text(parts))} words",
                  {"subject": parts["subject"], "body": body, "parts": parts}, label=label6)

        label7 = f"Check draft ({tag}v{version}, attempt {attempt})"
        em.start(7, label7)
        checks = check_draft(p, sig, parts, body)
        failed = [c for c in checks if c["passed"] is False]
        errored = [c for c in checks if c["passed"] is None]
        store.update_draft(draft_id, check_results=checks)
        if failed:
            summary = f"{len(failed)} failed: {', '.join(c['check'] for c in failed)}"
        elif errored:
            summary = f"{', '.join(c['check'] for c in errored)} could not run"
        else:
            summary = f"all {len(checks)} checks passed"
        em.finish(7, "warning" if failed or errored else "done", summary, {"checks": checks}, label=label7)
        if not failed and not errored:
            store.update_draft(draft_id, status="drafted")
            return draft_id, "drafted"
        if not failed:
            # A check could not run (e.g. LLM outage). Redrafting would not help; flag for the rep.
            break
        failures = [f"{c['check']}: {c['detail']}" for c in failed]
        previous = parts
    store.update_draft(draft_id, status="flagged")
    return draft_id, "flagged"


# =============================================================================
# One research round = Stages 3-5
# =============================================================================
SIGNAL_TABLE_COLS = ("id", "rank", "round", "type", "level", "summary", "date", "R", "R_reason", "T", "P",
                     "score", "passed_gate")


@step('research round')
def research_round(run_id, p, em, rnd, queries, state, more_rounds):
    """Search (new URLs only), rank, fetch full text, extract, merge into earlier
    signals, rate the new ones and re-rank all. Updates `state` (seen URLs,
    signals, kept count) and returns (passing signals best first, round record)."""
    sfx = "" if rnd == 1 else f" (round {rnd})"

    em.start(3, STAGE_NAMES[3] + sfx)
    sources, info, errors = gather_sources(queries, p["notes"] if rnd == 1 else "", exclude=state["seen"])
    cut = info.pop("cut")
    fetched = fetch_full_text(sources, p) if C.FETCH_TOP_N else {}
    state["seen"] |= {norm_url(s["url"]) for s in sources + cut}
    n_search = sum(s["origin"] == "search" for s in sources)
    state["kept"] += n_search
    store.add_sources(run_id, [{**s, "round": rnd} for s in sources + cut])  # cut too (kept=0), to tune ranking
    n_full = sum(v == "ok" for v in fetched.values())
    summary = (f"{len(queries)} queries -> {info['gathered']} {'new ' if rnd > 1 else ''}results, "
               f"kept top {n_search}; full text {n_full}/{len(fetched)}")
    if info["ranking"] and info["ranking"] != "embedding":
        summary += " (keyword ranking: embedding failed)"
    if errors:
        summary += f" ({len(errors)} queries failed)"
    em.finish(3, "warning" if errors or info["ranking"] != "embedding" else "done", summary,
              {**info, "errors": errors, "fetched": fetched, "sources": sources, "cut": cut})

    em.start(4, STAGE_NAMES[4] + sfx)
    new, dropped, grown = extract_signals(p, sources, existing=state["signals"])
    for s in dropped:
        s["source_urls"] = [s.pop("source_url")]
        s["verified"], s["round"] = 0, rnd
        s["id"] = store.add_signal(run_id, s)
    for s in new:
        s["verified"], s["round"] = 1, rnd
        s["id"] = store.add_signal(run_id, s)
    for s in grown:  # a new report can make the event date earlier: rescore now (R is unchanged)
        compute_score(s)
        store.update_signal(s["id"], source_urls=s["source_urls"], date=s["date"],
                            **{k: s[k] for k in ("T", "P", "score", "passed_gate")})
    em.finish(4, "done", f"{len(new) + len(dropped)} new signals, {len(new)} verified"
                         f"{f', {len(dropped)} dropped' if dropped else ''}"
                         f"{f', {len(grown)} merged into earlier signals' if grown else ''}",
              {"verified": new, "dropped": dropped, "merged": grown})

    em.start(5, STAGE_NAMES[5] + sfx)
    rate_signals(p, new)
    state["signals"].extend(new)
    ranked = rank_signals(state["signals"])
    for s in ranked:
        store.update_signal(s["id"], **{k: s[k] for k in ("R", "R_reason", "T", "P", "score", "passed_gate", "rank")})
    passing = [s for s in ranked if s["passed_gate"]]
    table = [{k: s.get(k) for k in SIGNAL_TABLE_COLS} for s in ranked]
    if passing:
        chosen, runner_ups = passing[0], passing[1:1 + C.N_RUNNER_UPS]
        em.finish(5, "done", f"Hook: {chosen['type']} ({chosen['level']}, round {chosen['round']}), "
                             f"score {chosen['score']:.2f}",
                  {"ranked": table, "chosen_id": chosen["id"], "runner_up_ids": [s["id"] for s in runner_ups]})
    else:
        nxt = f"research round {rnd + 1}" if more_rounds else "needs_research"
        em.finish(5, "warning", f"no signal with R >= {C.GATE_MIN_R} -> {nxt}", {"ranked": table})
    record = {"round": rnd, "queries": len(queries), "results": info["gathered"], "kept": n_search,
              "full_text": n_full, "new_signals": len(new), "merged": len(grown), "dropped": len(dropped),
              "passing": len(passing)}
    return passing, record


@step('plan queries')
def plan_round(p, rnd, state):
    """Stage 2 for round `rnd`. Returns (queries, method, extra payload)."""
    if rnd == 1:
        return plan_queries(p), "templates", {}
    gaps = research_gaps(state["signals"], state["kept"])
    if rnd == 2:
        return gap_queries(p, gaps, {_query_norm(q) for q in state["queries"]}), "gap templates", {"gaps": gaps}
    accepted, rejected = llm_queries(p, state["queries"], state["signals"], gaps)
    return accepted, "LLM", {"gaps": gaps, "rejected": rejected}


# =============================================================================
# Full pipeline
# =============================================================================
@step('prospect')
def run_pipeline(prospect, on_event=None):
    """Runs Stages 1-7. Returns the run id. Final status is one of
    drafted / generic / flagged / needs_research / failed ('generic' only when the
    Dashboard's generic-fallback switch is on and no signal passed the gate)."""
    p = {k: (prospect.get(k) or "").strip() for k in ("name", "company", "role", "website", "notes")}
    missing = validate_input(p)
    if missing:
        raise ValueError(f"required fields missing: {', '.join(missing)}")

    t_run = time.time()
    run_id = store.create_run(p)
    em = Emitter(run_id, on_event)
    obs.bind_run(run_id)
    obs.log.info("run started company=%s name=%s role=%s", p["company"], p["name"] or "-", p["role"] or "-")
    tag_current(run_id=run_id, company=p["company"], prospect=p["name"] or None, role=p["role"] or None)

    def end(status, error=None):
        store.update_run(run_id, status=status, duration=round(time.time() - t_run, 1), error=error)
        tag_current(status=status, error=error)
        obs.save_run_summary(run_id, status)
        return run_id

    stage = 1
    try:
        em.start(1)
        em.finish(1, "done", f"{p['name'] or '(name to find)'}, {p['role'] or '(role to find)'} at {p['company']}",
                  {k: v for k, v in p.items() if v})

        if not p["name"] or not p["role"]:
            field = "name" if not p["name"] else "role"
            label = f"Find prospect {field}"
            em.start(1, label)
            value, info = lookup_prospect(p, field)
            store.update_run(run_id, resolved=lookup_record(info))
            if value is None:
                target = (f"the {p['role']} of {p['company']}" if field == "name"
                          else f"{p['name']}'s role at {p['company']}")
                msg = f"Could not find {target} in the search results. Add the {field} and run again."
                em.finish(1, "warning", msg, info, label=label)
                return end("needs_research", msg)
            p[field] = value
            store.update_run(run_id, **{field: value})
            best, others = info["candidates"][0], [c["value"] for c in info["candidates"][1:]]
            em.finish(1, "warning" if others else "done",
                      f"{field}: {value} ({len(best['sources'])} source{'s' * (len(best['sources']) != 1)})"
                      + (f"; other candidates: {', '.join(others)}" if others else ""),
                      info, label=label)

        # Stages 2-5, repeated as research rounds until a signal passes the gate.
        state = {"seen": set(), "signals": [], "queries": [], "kept": 0}
        research, passing, stop_reason = [], [], None
        for rnd in range(1, max(1, C.MAX_RESEARCH_ROUNDS) + 1):
            more = rnd < C.MAX_RESEARCH_ROUNDS
            try:
                stage = 2
                em.start(2, STAGE_NAMES[2] + ("" if rnd == 1 else f" (round {rnd})"))
                queries, method, extra = plan_round(p, rnd, state)
                rej = extra.get("rejected") or []
                em.finish(2, "done" if queries else "warning",
                          f"{len(queries)} queries ({method})" + (f", {len(rej)} rejected" if rej else "")
                          + (f"; gaps: {', '.join(extra['gaps'])}" if extra.get("gaps") else ""),
                          {"queries": queries, **extra})
                if not queries:
                    stop_reason = f"round {rnd}: no new queries to run"
                    research.append({"round": rnd, "method": method, "queries": 0, **extra})
                    break
                state["queries"] += [q["query"] for q in queries]
                stage = 3
                passing, record = research_round(run_id, p, em, rnd, queries, state, more)
                research.append({**record, "method": method, **({"gaps": extra["gaps"]} if "gaps" in extra else {})})
            except Exception as e:  # StageError, or a bug in a retry round
                if rnd == 1:
                    raise  # nothing to fall back on: the run fails as before
                err = str(e) if isinstance(e, StageError) else f"{type(e).__name__}: {e}"
                em.fail_open(err)  # a later round failing keeps the run alive with round-1 results
                stop_reason = f"round {rnd} failed: {err}"
                research.append({"round": rnd, "error": err})
                break
            finally:
                # Rounds that completed a search (a round with no queries or that failed is in `research` only).
                store.update_run(run_id, rounds=sum("results" in x for x in research), research=research)
            if passing:
                break

        if not passing:
            msg = (f"No signal reached R >= {C.GATE_MIN_R} after {sum('results' in x for x in research)} "
                   f"research round(s) "
                   f"({state['kept']} sources, {len(state['signals'])} signals)."
                   + (f" {stop_reason[0].upper()}{stop_reason[1:]}." if stop_reason else "")
                   + " Add notes or a website and run again.")
            if not generic_fallback_on():
                return end("needs_research", msg)
            # Generic fallback (Dashboard switch on): a role-based draft, clearly marked as not personalised.
            stage = 6
            _, status = draft_and_check(run_id, p, None, em, version=1)
            return end("generic" if status == "drafted" else "flagged",
                       "Generic draft: no personalised hook was found. " + msg)
        chosen = passing[0]
        store.update_run(run_id, chosen_signal_id=chosen["id"])

        stage = 6
        _, status = draft_and_check(run_id, p, chosen, em, version=1)
        return end(status)

    except StageError as e:
        open_stages = em.fail_open(str(e))  # stage 6 closes its own event before raising
        return end("failed", f"stage {(open_stages or [stage])[-1]}: {e}")
    except Exception as e:  # unexpected bug: still close the run cleanly
        msg = f"{type(e).__name__}: {e}"
        open_stages = em.fail_open(msg)
        return end("failed", f"stage {(open_stages or [stage])[-1]}: {msg}")
    except BaseException:
        # Interrupted (Ctrl-C, or Streamlit rerun/stop when the page is touched mid-run).
        # Never leave the run stuck in 'running'.
        em.fail_open("interrupted")
        end("failed", "interrupted before finishing")
        raise


# =============================================================================
# Stage 8 - Review actions
# =============================================================================
def runner_ups(run_id):
    """The chosen hook plus the N_RUNNER_UPS kept at Stage 5, best first."""
    return [s for s in store.get_signals(run_id) if s["passed_gate"]][:1 + C.N_RUNNER_UPS]


def _reviewable(run):
    if run["status"] not in REVIEWABLE:
        raise ValueError(f"run {run['id']} is '{run['status']}', not reviewable")


def approve(run_id, subject=None, body=None):
    run = store.get_run(run_id)
    _reviewable(run)
    d = store.current_draft(run_id)
    if subject is not None or body is not None:
        save_edit(run_id, subject, body)
        d = store.get_draft(d["id"])
    store.update_draft(d["id"], status="approved")
    store.update_run(run_id, status="approved")
    store.add_review_action(run_id, d["id"], "approve")
    obs.log.info("review run=%s action=approve", run_id)
    return final_text(d)


def save_edit(run_id, subject, body):
    run = store.get_run(run_id)
    _reviewable(run)
    if not (subject or "").strip() or not (body or "").strip():
        raise ValueError("subject and body cannot be empty")
    d = store.current_draft(run_id)
    cur_subject = _pick(d["edited_subject"], d["subject"])
    cur_body = _pick(d["edited_body"], d["body"])
    if subject == cur_subject and body == cur_body:
        return False
    store.update_draft(d["id"], edited_subject=subject, edited_body=body)
    store.add_review_action(run_id, d["id"], "edit")
    return True


def discard(run_id, reason):
    run = store.get_run(run_id)
    _reviewable(run)
    d = store.current_draft(run_id)
    store.update_draft(d["id"], status="discarded")
    store.update_run(run_id, status="discarded")
    store.add_review_action(run_id, d["id"], "discard", reason)
    obs.log.info("review run=%s action=discard reason=%s", run_id, reason)


@step('regenerate')
def regenerate(run_id, signal_id=None, on_event=None):
    """Redraft from the same signal, or from a runner-up the rep picks.
    Returns (status, error). If drafting fails, the old draft stays current and
    the run keeps its previous status, so the rep can still act on it."""
    run = store.get_run(run_id)
    _reviewable(run)
    old = store.current_draft(run_id)
    if signal_id is None and run["chosen_signal_id"] is None:
        sig = None  # a generic run (no signal passed the gate): redraft the generic email
    else:
        sig = store.get_signal(signal_id or run["chosen_signal_id"])
        allowed = {s["id"] for s in runner_ups(run_id)}
        if sig is None or sig["id"] not in allowed:
            raise ValueError("signal must be the chosen hook or one of its runner-ups")
    sig_id = sig["id"] if sig else None
    p = {k: run[k] for k in ("name", "company", "role", "website", "notes")}
    store.add_review_action(run_id, old["id"], "regenerate", f"signal {sig_id}" if sig else "generic")
    version = max(d["version"] for d in store.get_drafts(run_id)) + 1
    store.update_run(run_id, status="running")
    em = Emitter(run_id, on_event)
    obs.bind_run(run_id)
    obs.log.info("regenerate signal=%s", sig_id if sig else "generic")
    t = time.time()

    def restore(err):
        em.fail_open(err)
        store.update_run(run_id, status=run["status"], chosen_signal_id=run["chosen_signal_id"],
                         error=f"regenerate failed: {err}")
        store.add_review_action(run_id, old["id"], "regenerate_failed", err)

    try:
        _, status = draft_and_check(run_id, p, sig, em, version, replaces=old["id"])
    except StageError as e:
        restore(str(e))
        return run["status"], str(e)
    except Exception as e:
        restore(f"{type(e).__name__}: {e}")
        return run["status"], f"{type(e).__name__}: {e}"
    except BaseException:
        cur = store.current_draft(run_id)
        if cur and cur["id"] != old["id"]:  # a new draft was written before the interrupt
            store.update_draft(cur["id"], status="flagged")
            store.update_run(run_id, status="flagged", chosen_signal_id=sig_id)
            em.fail_open("interrupted")
        else:
            restore("interrupted")
        raise
    if sig is None and status == "drafted":
        status = "generic"
    store.update_run(run_id, status=status, chosen_signal_id=sig_id,
                     error=run["error"] if sig is None else None,  # keep the "why generic" note
                     duration=round((run["duration"] or 0) + time.time() - t, 1))
    obs.save_run_summary(run_id, status)
    return status, None


def _pick(edited, original):
    return edited if edited is not None else original


def final_text(d):
    return f"Subject: {_pick(d['edited_subject'], d['subject'])}\n\n{_pick(d['edited_body'], d['body'])}"


# =============================================================================
# CLI
# =============================================================================
def _print_event(ev):
    if ev["status"] == "started":
        print(f"[{ev['stage']}] {ev['stage_name']} ...", flush=True)
    else:
        print(f"[{ev['stage']}] {ev['stage_name']}: {ev['status'].upper()} - {ev['summary']} "
              f"({ev['elapsed']}s)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Run the outreach pipeline for one prospect. Give --name, "
                                             "--role or both; the missing one is looked up by search.")
    ap.add_argument("company")
    ap.add_argument("--name", default="")
    ap.add_argument("--role", default="")
    ap.add_argument("--website", default="")
    ap.add_argument("--notes", default="")
    a = ap.parse_args()
    if validate_input(vars(a)):
        ap.error(f"required: {', '.join(validate_input(vars(a)))}")
    rid = run_pipeline(vars(a), on_event=_print_event)
    run = store.get_run(rid)
    print(f"\nRun {rid}: {run['status']}" + (f" ({run['error']})" if run["error"] else ""))
    res = run.get("resolved")
    if res and res.get("value"):
        print(f"  {res['field']} found by search (confirm before sending): {res['value']} "
              f"<- {', '.join(res['sources'])}")
    for s in store.get_signals(rid):
        if s["verified"]:
            print(f"  #{s['rank']} score={s['score']} R={s['R']} T={s['T']} P={s['P']} "
                  f"[{s['type']}/{s['level']}] {s['summary']}")
    d = store.current_draft(rid)
    if d:
        print("\n" + final_text(d))
        for c in d["check_results"] or []:
            print(f"  {'PASS' if c['passed'] else 'FAIL'} {c['check']}: {c['detail']}")
