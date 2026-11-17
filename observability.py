"""Observability: what the pipeline did, how long it took, what failed, and how good the output was.

Three layers (see OBSERVABILITY.md for the full plan and the list of observations):

1. Structured logs   Python `logging`, every line tagged with run id and stage.
                     Level from env OUTREACH_LOG_LEVEL (default INFO).
2. Call records      One row per LLM attempt (`llm_calls`) and per tool call
                     (`tool_calls`: search, fetch, embedding) in SQLite, tagged
                     with run id and stage. A per-run summary is saved in runs.obs.
3. Report            `report()` aggregates the records plus the existing tables
                     (drafts, review_actions, runs) for the Dashboard "Health" page.

Recording never raises: a bug here must not fail a run. The run id and stage travel in
context variables; tracing.Pool copies them into worker threads.
"""

import contextvars
import difflib
import logging
import os
import time
from datetime import datetime, timezone

import pandas as pd

import config as C
import store

_run = contextvars.ContextVar("obs_run", default=None)
_stage = contextvars.ContextVar("obs_stage", default=None)

log = logging.getLogger("outreach")


# --- context ---------------------------------------------------------------------
def bind_run(run_id):
    _run.set(run_id)
    _stage.set(None)


def bind_stage(stage):
    _stage.set(stage)


def current_run():
    return _run.get()


# --- logging ---------------------------------------------------------------------
class _Ctx(logging.Filter):
    def filter(self, record):
        record.run, record.stage = _run.get() or "-", _stage.get() or "-"
        return True


def setup_logging():
    """Idempotent. Adds one stderr handler to the 'outreach' logger."""
    if getattr(log, "_configured", False):
        return
    h = logging.StreamHandler()
    h.addFilter(_Ctx())
    h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s run=%(run)s stage=%(stage)s %(message)s"))
    log.addHandler(h)
    log.setLevel(os.getenv("OUTREACH_LOG_LEVEL", "INFO").upper())
    log.propagate = False
    log._configured = True


setup_logging()


# --- recording -------------------------------------------------------------------
def _safe(fn):
    def wrapper(*a, **kw):
        try:
            return fn(*a, **kw)
        except Exception as e:  # never fail the pipeline because of observability
            try:
                log.warning("observability record failed: %s: %s", type(e).__name__, e)
            except Exception:
                pass
    return wrapper


@_safe
def record_llm(**rec):
    """purpose, model, attempt, outcome (ok|retry|error|deadline), latency, tokens, error."""
    rec.setdefault("run_id", _run.get())
    rec.setdefault("stage", _stage.get())
    store.add_llm_call(rec)
    lvl = logging.INFO if rec.get("outcome") == "ok" else logging.WARNING
    log.log(lvl, "llm %s model=%s attempt=%s outcome=%s %.1fs tokens=%s/%s%s", rec.get("purpose"),
            rec.get("model"), rec.get("attempt"), rec.get("outcome"), rec.get("latency") or 0,
            rec.get("input_tokens"), rec.get("output_tokens"),
            f" error={rec['error']}" if rec.get("error") else "")


@_safe
def record_tool(**rec):
    """tool (search|fetch|embedding), backend, outcome (ok|empty|error|timeout), latency, n_results."""
    rec.setdefault("run_id", _run.get())
    rec.setdefault("stage", _stage.get())
    store.add_tool_call(rec)
    lvl = logging.INFO if rec.get("outcome") in ("ok", "empty") else logging.WARNING
    log.log(lvl, "%s backend=%s outcome=%s %.1fs n=%s fallbacks=%s %s", rec.get("tool"), rec.get("backend"),
            rec.get("outcome"), rec.get("latency") or 0, rec.get("n_results"), rec.get("fallbacks"),
            rec.get("detail") or "")


def usage_callback():
    """A LangChain callback that collects token usage of one model call, or None if
    langchain-core is missing. Read it afterwards from .input_tokens / .output_tokens."""
    try:
        from langchain_core.callbacks import BaseCallbackHandler
    except Exception:
        return None

    class Usage(BaseCallbackHandler):
        input_tokens = output_tokens = None

        def on_llm_end(self, response, **kwargs):
            try:
                for gens in response.generations:
                    for g in gens:
                        u = getattr(getattr(g, "message", None), "usage_metadata", None) or {}
                        if u:
                            self.input_tokens = (self.input_tokens or 0) + (u.get("input_tokens") or 0)
                            self.output_tokens = (self.output_tokens or 0) + (u.get("output_tokens") or 0)
            except Exception:
                pass

    return Usage()


@_safe
def save_run_summary(run_id, status):
    """Roll the run's call records up into runs.obs and log one summary line."""
    s = run_summary(run_id)
    store.update_run(run_id, obs=s)
    log.info("run done status=%s llm_calls=%s retries=%s tokens=%s/%s llm_time=%.0fs tool_failures=%s",
             status, s["llm"]["attempts"], s["llm"]["retries"], s["llm"]["input_tokens"],
             s["llm"]["output_tokens"], s["llm"]["time_s"], s["tools"]["failed"])


def run_summary(run_id):
    llm = pd.DataFrame(store.get_llm_calls(run_id))
    tools = pd.DataFrame(store.get_tool_calls(run_id))
    out = {"llm": {"attempts": 0, "retries": 0, "failed": 0, "input_tokens": 0, "output_tokens": 0,
                   "time_s": 0.0, "models": {}},
           "tools": {"calls": 0, "failed": 0, "by_tool": {}}}
    if not llm.empty:
        out["llm"].update(attempts=len(llm), retries=int((llm["attempt"] > 0).sum()),
                          failed=int((llm["outcome"] != "ok").sum()),
                          input_tokens=int(_num(llm["input_tokens"]).sum()),
                          output_tokens=int(_num(llm["output_tokens"]).sum()),
                          time_s=round(float(_num(llm["latency"]).sum()), 1),
                          models={str(k): int(v) for k, v in llm["model"].value_counts().items()})
    if not tools.empty:
        bad = ~tools["outcome"].isin(["ok", "empty"])
        out["tools"].update(calls=len(tools), failed=int(bad.sum()),
                            by_tool={str(k): int(v) for k, v in tools["tool"].value_counts().items()})
    return out


# =============================================================================
# Report (Dashboard "Health" page)
# =============================================================================
def _num(col):
    """Column as numbers with gaps as 0 (token counts are None when the SDK reports no usage)."""
    return pd.to_numeric(col, errors="coerce").fillna(0)


def _pct(s, q):
    s = s.dropna()
    return round(float(s.quantile(q)), 1) if len(s) else None


def _rate(num, den):
    return round(100 * num / den, 1) if den else None


def _llm_table(df, by):
    rows = []
    for key, g in df.groupby(by):
        rows.append({by: key, "attempts": len(g), "failed %": _rate(int((g["outcome"] != "ok").sum()), len(g)),
                     "retried %": _rate(int((g["outcome"] == "retry").sum()), len(g)),
                     "p50 s": _pct(g["latency"], .5), "p95 s": _pct(g["latency"], .95),
                     "in tokens": int(_num(g["input_tokens"]).sum()),
                     "out tokens": int(_num(g["output_tokens"]).sum())})
    return pd.DataFrame(rows)


def _tool_table(df):
    rows = []
    for (tool, backend), g in df.groupby(["tool", df["backend"].fillna("-")]):
        n = len(g)
        rows.append({"tool": tool, "backend": backend, "calls": n,
                     "failed %": _rate(int((~g["outcome"].isin(["ok", "empty"])).sum()), n),
                     "empty %": _rate(int((g["outcome"] == "empty").sum()), n),
                     "p50 s": _pct(g["latency"], .5), "p95 s": _pct(g["latency"], .95)})
    return pd.DataFrame(rows)


def _quality(runs, drafts):
    """First-attempt check pass rates, redraft rate and edit size."""
    out = {"checks": pd.DataFrame(), "redraft_rate": None, "edit_size": None}
    if drafts.empty:
        return out
    first = drafts[(drafts["attempt"] == 1) & (drafts["version"] == 1)]
    rows = {}
    for kind, g in first.groupby(first["kind"].fillna("personalised")):
        for res in g["check_results"].dropna():
            for c in res:
                r = rows.setdefault((c["check"], kind), [0, 0, 0])
                r[0 if c["passed"] is True else 1 if c["passed"] is False else 2] += 1
    out["checks"] = pd.DataFrame([{"check": k[0], "draft kind": k[1], "pass": v[0], "fail": v[1],
                                   "could not run": v[2], "pass %": _rate(v[0], v[0] + v[1])}
                                  for k, v in sorted(rows.items())])
    out["redraft_rate"] = _rate(int(((drafts["attempt"] == 2) & (drafts["version"] == 1)).sum()), len(first))
    edited = drafts.dropna(subset=["edited_body"])
    if len(edited):
        out["edit_size"] = round(float(pd.Series(
            [1 - difflib.SequenceMatcher(None, d["body"] or "", d["edited_body"]).ratio()
             for _, d in edited.iterrows()]).median()) * 100, 1)
    return out


def _funnel(runs):
    rows = []
    for _, r in runs.iterrows():
        for rec in r["research"] or []:
            if "results" in rec:
                rows.append({k: rec.get(k, 0) for k in ("results", "kept", "full_text", "new_signals",
                                                         "dropped", "passing")} | {"round": rec["round"]})
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    return df.groupby("round").mean().round(1).reset_index().rename(columns={
        "results": "results gathered", "kept": "sources kept", "full_text": "pages fetched ok",
        "new_signals": "signals extracted", "dropped": "signals dropped (unverified)",
        "passing": "signals passing R gate"})


def stuck_runs(runs):
    if runs.empty:
        return runs
    age = (datetime.now(timezone.utc) - pd.to_datetime(runs["created_at"], utc=True)).dt.total_seconds() / 60
    return runs[(runs["status"] == "running") & (age > C.OBS_STUCK_MINUTES)]


def report():
    runs = pd.DataFrame(store.list_runs())
    llm = pd.DataFrame(store.get_llm_calls())
    tools = pd.DataFrame(store.get_tool_calls())
    drafts = pd.DataFrame(store.all_drafts())
    actions = pd.DataFrame(store.all_review_actions())
    rep = {"n_runs": len(runs), "llm_by_model": pd.DataFrame(), "llm_by_purpose": pd.DataFrame(),
           "llm_errors": pd.DataFrame(), "tools": pd.DataFrame(), "tool_errors": pd.DataFrame(),
           "funnel": pd.DataFrame(), "status_mix": pd.DataFrame(), "failure_stages": pd.DataFrame(),
           "review": pd.DataFrame(), "stuck": pd.DataFrame(), "tokens_per_run": None,
           "quality": {"checks": pd.DataFrame(), "redraft_rate": None, "edit_size": None},
           "overall": {}}
    if not llm.empty:
        rep["overall"]["llm_attempts"] = len(llm)
        rep["overall"]["llm_error_pct"] = _rate(int((llm["outcome"] != "ok").sum()), len(llm))
        rep["overall"]["llm_retry_attempt_pct"] = _rate(int((llm["attempt"] > 0).sum()), len(llm))
        rep["llm_by_model"] = _llm_table(llm, "model")
        rep["llm_by_purpose"] = _llm_table(llm, "purpose")
        bad = llm[llm["outcome"] != "ok"]
        if len(bad):
            rep["llm_errors"] = (bad.assign(error=bad["error"].fillna("").str[:90])
                                 .groupby(["outcome", "error"]).size().reset_index(name="count")
                                 .sort_values("count", ascending=False))
        tok = llm.assign(input_tokens=_num(llm["input_tokens"]), output_tokens=_num(llm["output_tokens"]))
        per_run = tok.groupby("run_id")[["input_tokens", "output_tokens"]].sum()
        rep["tokens_per_run"] = {"in": int(per_run["input_tokens"].median()),
                                 "out": int(per_run["output_tokens"].median())}
    if not tools.empty:
        rep["tools"] = _tool_table(tools)
        srch = tools[tools["tool"] == "search"]
        if len(srch):
            rep["overall"]["search_calls"] = len(srch)
            rep["overall"]["search_failed_pct"] = _rate(int((srch["outcome"] == "error").sum()
                                                            + (srch["outcome"] == "timeout").sum()), len(srch))
            rep["overall"]["search_empty_pct"] = _rate(int((srch["outcome"] == "empty").sum()), len(srch))
        bad = tools[~tools["outcome"].isin(["ok", "empty"])]
        if len(bad):
            rep["tool_errors"] = (bad.assign(detail=bad["detail"].fillna("").str[:90])
                                  .groupby(["tool", "backend", "detail"]).size().reset_index(name="count")
                                  .sort_values("count", ascending=False))
    if not runs.empty:
        rep["status_mix"] = runs["status"].value_counts().rename_axis("status").reset_index(name="runs")
        rep["funnel"] = _funnel(runs)
        rep["stuck"] = stuck_runs(runs)
        failed = runs[runs["status"] == "failed"]
        if len(failed):  # error text starts "stage N: ..."
            st = failed["error"].fillna("").str.extract(r"^(stage \d+)")[0].fillna("other")
            rep["failure_stages"] = st.value_counts().rename_axis("where").reset_index(name="runs")
    if not actions.empty:
        rep["review"] = actions["action"].value_counts().rename_axis("action").reset_index(name="count")
    rep["quality"] = _quality(runs, drafts)
    return rep


def alerts(rep):
    """Threshold breaches, as sentences. Thresholds are the starting defaults in config.OBS_ALERTS."""
    T, o, out = C.OBS_ALERTS, rep["overall"], []
    checks = [("llm_error_pct", "llm_error_pct", "LLM attempt error rate"),
              ("retry_attempt_pct", "llm_retry_attempt_pct",
               "LLM attempts that were retries (primary model overloaded or slow?)"),
              ("search_failed_pct", "search_failed_pct", "Search calls where every backend failed"),
              ("search_empty_pct", "search_empty_pct", "Search calls with no results")]
    for tkey, okey, label in checks:
        if o.get(okey) is not None and o[okey] > T[tkey]:
            out.append(f"{label}: {o[okey]}% > {T[tkey]}%")
    mix = rep["status_mix"]
    if not mix.empty and rep["n_runs"] >= T["min_runs_for_rate_alerts"]:
        for s, key in (("failed", "run_failed_pct"), ("needs_research", "needs_research_pct")):
            v = _rate(int(mix.loc[mix["status"] == s, "runs"].sum()), rep["n_runs"])
            if v is not None and v > T[key]:
                out.append(f"Runs with status {s}: {v}% > {T[key]}%")
    if len(rep["stuck"]):
        out.append(f"{len(rep['stuck'])} run(s) stuck in 'running' for over {C.OBS_STUCK_MINUTES} min")
    return out


class timer:
    """with timer() as t: ...; t.s  (seconds)"""

    def __enter__(self):
        self.t0 = time.time()
        return self

    def __exit__(self, *exc):
        self.s = round(time.time() - self.t0, 3)
        return False
