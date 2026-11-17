"""Eval metrics: turn the records of one eval run (an isolated SQLite DB) into numbers.

    compute(db_path, cases) -> dict   (JSON-serialisable, nested)
    flatten(metrics)        -> {"latency.end_to_end_s.mean": 61.2, ...}   (used by compare)
    to_markdown(...)        -> report text
    per_case_frame(...)     -> one row per case, for CSV

`cases` = [{"case_id", "category", "expect_status", "run_id" (None if the harness
failed before a run existed), "wall_s", "harness_error"}], in dataset order. Only the
runs listed there are counted; calls from any other run in the DB are ignored.
"""

import math
import re
from collections import defaultdict

import pandas as pd

import config as C
import store

# Pipeline components in execution order, and the LLM purpose that belongs to each.
COMPONENTS = ["Input", "Find prospect", "Plan queries", "Gather sources", "Extract signals",
              "Score & pick hook", "Draft", "Check draft"]
PURPOSE_COMPONENT = {"Lookup": "Find prospect", "QueryPlan": "Plan queries", "Extraction": "Extract signals",
                     "Ratings": "Score & pick hook", "DraftParts": "Draft", "Grounding": "Check draft"}
LOOKUP_CATEGORIES = {"name_only", "role_only"}  # rows where Stage 1 must look something up


def component_of(stage_name):
    """'Gather sources (round 2)' -> 'Gather sources'; 'Find prospect role' -> 'Find prospect'."""
    base = re.sub(r"\s*\(.*\)\s*$", "", stage_name or "").strip()
    return "Find prospect" if base.startswith("Find prospect") else base


# --- small helpers --------------------------------------------------------------
def _f(x, nd=2):
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else round(float(x), nd)


def stats(values, nd=2):
    s = pd.Series([v for v in values if v is not None], dtype=float).dropna()
    if s.empty:
        return {"n": 0}
    return {"n": int(len(s)), "mean": _f(s.mean(), nd), "p50": _f(s.quantile(.5), nd),
            "p95": _f(s.quantile(.95), nd), "max": _f(s.max(), nd)}


def pct(num, den):
    return round(100 * num / den, 1) if den else None


def _num(col):
    return pd.to_numeric(col, errors="coerce")


def _price(model, tin, tout):
    p = getattr(C, "MODEL_PRICE_PER_1M", {}).get(model)
    return None if p is None else (tin * p[0] + tout * p[1]) / 1e6


# --- per-case extraction -----------------------------------------------------------
def _case_record(case, run, llm, tools, drafts):
    """Everything measured for one case, flat enough to become a CSV row."""
    rec = {"case_id": case["case_id"], "category": case["category"], "expect_status": case.get("expect_status") or "",
           "run_id": case.get("run_id"), "wall_s": case.get("wall_s"), "harness_error": case.get("harness_error") or ""}
    if run is None:
        rec["status"] = "harness_error"
        return rec
    rec.update(status=run["status"], duration_s=run["duration"], rounds=run.get("rounds") or 0,
               error=run.get("error") or "")
    exp = rec["expect_status"]
    rec["expect_ok"] = None if not exp else run["status"] == exp

    comp = defaultdict(float)
    for ev in store.get_events(run["id"]):
        if ev["elapsed"] is not None:
            comp[component_of(ev["stage_name"])] += ev["elapsed"]
    for k in COMPONENTS:
        rec[f"lat_{k}"] = _f(comp[k], 2) if k in comp else None
    rec["lat_sum_components"] = _f(sum(comp.values()), 2)
    rec["lat_unaccounted"] = _f((run["duration"] or 0) - sum(comp.values()), 2) if run["duration"] is not None else None

    ll = llm[llm["run_id"] == run["id"]]
    rec["llm_attempts"] = int(len(ll))
    rec["llm_retries"] = int((ll["attempt"] > 0).sum()) if len(ll) else 0
    rec["tokens_in"] = int(_num(ll["input_tokens"]).fillna(0).sum()) if len(ll) else 0
    rec["tokens_out"] = int(_num(ll["output_tokens"]).fillna(0).sum()) if len(ll) else 0
    tl = tools[tools["run_id"] == run["id"]]
    rec["search_calls"] = int((tl["tool"] == "search").sum()) if len(tl) else 0
    rec["tool_failures"] = int((~tl["outcome"].isin(["ok", "empty"])).sum()) if len(tl) else 0

    res = run.get("research") or []
    for k, src in (("results", "results"), ("kept", "kept"), ("pages_ok", "full_text"), ("signals", "new_signals"),
                   ("dropped", "dropped"), ("passing", "passing")):
        rec[k] = sum(r.get(src, 0) for r in res if "results" in r)
    rec["win_round"] = next((r["round"] for r in res if r.get("passing")), None)

    sig = store.get_signal(run["chosen_signal_id"]) if run.get("chosen_signal_id") else None
    rec["hook_type"] = sig["type"] if sig else ("generic" if run["status"] in ("generic",) else None)
    rec["hook_score"] = _f(sig["score"], 3) if sig else None
    r = run.get("resolved")
    rec["lookup"] = None if not isinstance(r, dict) else ("found" if r.get("value") else "not_found")

    dr = drafts[drafts["run_id"] == run["id"]]
    first = dr[(dr["attempt"] == 1) & (dr["version"] == 1)]
    rec["redraft"] = bool(((dr["attempt"] == 2) & (dr["version"] == 1)).any()) if len(dr) else None
    unsupported = None
    for _, d in first.iterrows():
        for c in d["check_results"] or []:
            rec[f"check1_{c['check']}"] = c["passed"]
            if c["check"] == "grounding" and c.get("claims") is not None:
                unsupported = sum(1 for x in c["claims"] if not x.get("supported"))
    rec["unsupported_claims"] = unsupported
    cur = store.current_draft(run["id"])
    rec["body_words"] = len((cur["edited_body"] or cur["body"] or "").split()) if cur else None
    return rec


# --- aggregation ---------------------------------------------------------------------
def compute(db_path, cases, label=""):
    store.DB_PATH = db_path
    store.init_db()
    ids = {c["run_id"] for c in cases if c.get("run_id")}
    runs = {r["id"]: r for r in store.list_runs() if r["id"] in ids}
    llm = pd.DataFrame(store.get_llm_calls()).reindex(columns=["run_id", "stage", "purpose", "model", "attempt",
                                                                "outcome", "latency", "input_tokens", "output_tokens"])
    tools = pd.DataFrame(store.get_tool_calls()).reindex(columns=["run_id", "tool", "backend", "outcome", "latency",
                                                                   "n_results", "fallbacks"])
    drafts = pd.DataFrame(store.all_drafts()).reindex(columns=["run_id", "attempt", "version", "check_results"])
    llm, tools, drafts = (d[d["run_id"].isin(ids)] for d in (llm, tools, drafts))

    recs = [_case_record(c, runs.get(c.get("run_id")), llm, tools, drafts) for c in cases]
    df = pd.DataFrame(recs)
    n, ran = len(df), df[df["status"] != "harness_error"]
    m = {"label": label, "n_cases": n, "n_runs": int(len(ran))}

    # ---- outcome
    sc = df["status"].value_counts()
    m["outcome"] = {"status_counts": {k: int(v) for k, v in sc.items()},
                    "status_pct": {k: pct(v, n) for k, v in sc.items()}}
    for s in ("needs_research", "failed", "generic", "flagged", "drafted", "harness_error"):
        m["outcome"][f"{s}_pct"] = pct(int(sc.get(s, 0)), n)
    m["outcome"]["usable_draft_pct"] = pct(int(sc.reindex(["drafted", "generic", "flagged"]).fillna(0).sum()), n)
    m["outcome"]["by_category"] = {cat: {k: int(v) for k, v in g["status"].value_counts().items()}
                                   for cat, g in df.groupby("category")}
    ex = df[df["expect_status"] != ""]
    m["expectations"] = {"n": int(len(ex)), "pass_pct": pct(int((ex["expect_ok"] == True).sum()), len(ex))}  # noqa: E712
    posr = df[df["expect_status"] == ""]
    m["non_negative_cases"] = {"n": int(len(posr)),
                               "needs_research_pct": pct(int((posr["status"] == "needs_research").sum()), len(posr)),
                               "failed_pct": pct(int((posr["status"] == "failed").sum()), len(posr)),
                               "draft_pct": pct(int(posr["status"].isin(["drafted", "generic", "flagged"]).sum()),
                                                len(posr))}

    # ---- latency
    comp_stats, mean_sum = {}, 0.0
    for k in COMPONENTS:
        col = ran[f"lat_{k}"] if f"lat_{k}" in ran else pd.Series(dtype=float)
        st = stats(col.tolist())
        comp_stats[k] = st
        if st["n"]:
            # mean over ALL runs (0 when the component did not run) so the shares add up to the whole run
            mean_sum += float(col.fillna(0).sum()) / max(len(ran), 1)
    e2e = stats(ran["duration_s"].tolist()) if "duration_s" in ran else {"n": 0}
    for k, st in comp_stats.items():
        if st["n"] and e2e.get("mean"):
            st["share_of_e2e_pct"] = round(100 * float(ran[f"lat_{k}"].fillna(0).sum()) / max(len(ran), 1)
                                           / e2e["mean"], 1)
    ok_llm = llm[llm["outcome"] == "ok"]
    per_call = {"llm": {p: stats(g["latency"].tolist()) for p, g in ok_llm.groupby("purpose")},
                "search": {str(b): stats(g["latency"].tolist())
                           for b, g in tools[(tools["tool"] == "search") & (tools["outcome"] == "ok")].groupby("backend")},
                "fetch": stats(tools[(tools["tool"] == "fetch") & (tools["outcome"] == "ok")]["latency"].tolist()),
                "embedding": stats(tools[(tools["tool"] == "embedding") & (tools["outcome"] == "ok")]["latency"].tolist())}
    m["latency"] = {"end_to_end_s": e2e, "components_s": comp_stats,
                    "sum_of_component_means_s": _f(mean_sum),
                    "unaccounted_mean_s": _f(ran["lat_unaccounted"].mean()) if "lat_unaccounted" in ran else None,
                    "wall_per_case_s": stats(df["wall_s"].tolist()), "per_call_s": per_call}

    # ---- tokens (and cost, if MODEL_PRICE_PER_1M is set)
    ok_tok = ok_llm.assign(tin=_num(ok_llm["input_tokens"]), tout=_num(ok_llm["output_tokens"]))
    comp_tok = {}
    tot_in, tot_out = float(ok_tok["tin"].sum()), float(ok_tok["tout"].sum())
    for purpose, comp in PURPOSE_COMPONENT.items():
        g = ok_tok[ok_tok["purpose"] == purpose]
        if g.empty:
            continue
        runs_with = g["run_id"].nunique()
        comp_tok[comp] = {"calls": int(len(g)), "runs": int(runs_with),
                          "in_per_call": _f(g["tin"].mean(), 0), "out_per_call": _f(g["tout"].mean(), 0),
                          "in_per_run": _f(g["tin"].sum() / runs_with, 0), "out_per_run": _f(g["tout"].sum() / runs_with, 0),
                          "share_of_tokens_pct": pct(float(g["tin"].sum() + g["tout"].sum()), tot_in + tot_out)}
    per_run_tok = ran[["tokens_in", "tokens_out"]] if "tokens_in" in ran else pd.DataFrame()
    cost = None
    if getattr(C, "MODEL_PRICE_PER_1M", {}):
        cost = {"total_usd": 0.0, "models_without_price": sorted({str(x) for x in ok_tok["model"].unique()
                                                                   if x not in C.MODEL_PRICE_PER_1M})}
        for _, r in ok_tok.iterrows():
            c = _price(r["model"], r["tin"] or 0, r["tout"] or 0)
            cost["total_usd"] += c or 0
        cost["per_run_usd"] = _f(cost["total_usd"] / max(len(ran), 1), 5)
        cost["total_usd"] = _f(cost["total_usd"], 4)
    m["tokens"] = {"per_run_in": stats(per_run_tok["tokens_in"].tolist(), 0) if len(per_run_tok) else {"n": 0},
                   "per_run_out": stats(per_run_tok["tokens_out"].tolist(), 0) if len(per_run_tok) else {"n": 0},
                   "total_in": int(tot_in), "total_out": int(tot_out), "components": comp_tok,
                   "ok_attempts_without_token_count": int(ok_tok["input_tokens"].isna().sum()),
                   "cost": cost}

    # ---- reliability
    la = len(llm)
    m["reliability"] = {
        "llm_attempts": la,
        **{f"llm_{o}_pct": pct(int((llm["outcome"] == o).sum()), la) for o in ("ok", "retry", "error", "deadline", "empty")},
        "llm_attempts_per_run": _f(la / max(len(ran), 1)),
    }
    srch = tools[tools["tool"] == "search"]
    sok = srch[srch["outcome"] == "ok"]
    m["reliability"]["search"] = {
        "calls": int(len(srch)), "empty_pct": pct(int((srch["outcome"] == "empty").sum()), len(srch)),
        "failed_pct": pct(int(srch["outcome"].isin(["error", "timeout"]).sum()), len(srch)),
        "fallback_pct": pct(int((_num(sok["fallbacks"]) > 0).sum()), len(sok)),
        "backend_mix_pct": {str(k): pct(int(v), len(sok)) for k, v in sok["backend"].value_counts().items()}}
    fe = tools[tools["tool"] == "fetch"]
    m["reliability"]["fetch"] = {"calls": int(len(fe)), "ok_pct": pct(int((fe["outcome"] == "ok").sum()), len(fe)),
                                 "error_pct": pct(int((fe["outcome"] == "error").sum()), len(fe))}
    em = tools[tools["tool"] == "embedding"]
    m["reliability"]["embedding"] = {"calls": int(len(em)), "ok_pct": pct(int((em["outcome"] == "ok").sum()), len(em)),
                                     "keyword_fallback_pct": pct(int((em["outcome"] != "ok").sum()), len(em))}
    m["reliability"]["runs_with_any_tool_failure_pct"] = pct(int((ran["tool_failures"] > 0).sum()), len(ran)) if len(ran) else None

    # ---- research funnel, rounds, lookup
    fr = ran[ran["rounds"] > 0] if "rounds" in ran else ran
    m["funnel_mean_per_run"] = {k: _f(fr[k].mean(), 1) for k in ("results", "kept", "pages_ok", "signals", "dropped",
                                                                   "passing") if k in fr and len(fr)}
    m["rounds"] = {"distribution": {int(k): int(v) for k, v in fr["rounds"].value_counts().sort_index().items()} if len(fr) else {},
                   "needed_round2_plus_pct": pct(int((fr["rounds"] >= 2).sum()), len(fr)),
                   "winning_round": {int(k): int(v) for k, v in ran["win_round"].dropna().value_counts().sort_index().items()}
                   if "win_round" in ran else {},
                   "search_calls_per_run": _f(ran["search_calls"].mean(), 1) if len(ran) else None}
    lk = ran[ran["lookup"].notna()] if "lookup" in ran else ran.iloc[0:0]
    m["lookup"] = {"n": int(len(lk)), "found_pct": pct(int((lk["lookup"] == "found").sum()), len(lk))}
    hk = ran["hook_type"].dropna() if "hook_type" in ran else pd.Series(dtype=str)
    m["hooks"] = {"type_counts": {k: int(v) for k, v in hk.value_counts().items()},
                  "score": stats(ran["hook_score"].tolist(), 3) if "hook_score" in ran else {"n": 0}}

    # ---- draft quality (rule-based checks only; not a measure of how good an email is)
    dr_ran = ran[ran["body_words"].notna()] if "body_words" in ran else ran.iloc[0:0]
    chk = {}
    for col in [c for c in df.columns if c.startswith("check1_")]:
        s = df[col]
        done = s.isin([True, False])  # None = the check could not run (or there was no draft)
        chk[col.removeprefix("check1_")] = {"pass_pct": pct(int((s == True).sum()), int(done.sum())),  # noqa: E712
                                            "n": int(done.sum())}
    m["quality"] = {"drafts": int(len(dr_ran)), "first_attempt_checks": chk,
                    "redraft_pct": pct(int((dr_ran["redraft"] == True).sum()), len(dr_ran)) if len(dr_ran) else None,
                    "flagged_pct": pct(int((ran["status"] == "flagged").sum()), len(dr_ran)) if len(dr_ran) else None,
                    "body_words": stats(dr_ran["body_words"].tolist(), 0),
                    "unsupported_claims_per_first_draft": _f(df["unsupported_claims"].mean()) if "unsupported_claims" in df else None}
    return m, df


# --- output helpers ----------------------------------------------------------------
def flatten(d, prefix=""):
    """Nested metrics -> {"a.b.c": number}. Only numbers are kept (what compare diffs)."""
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, key + "."))
        elif isinstance(v, (int, float)) and not isinstance(v, bool) and v is not None:
            out[key] = v
    return out


def _tbl(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join("–" if c is None else str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def to_markdown(m, manifest):
    L, o, lat, tok, rel = [], m["outcome"], m["latency"], m["tokens"], m["reliability"]
    e = lat["end_to_end_s"]
    L += [f"# Eval report: {m['label']}", "",
          f"{m['n_cases']} cases, {m['n_runs']} produced a run. Dataset `{manifest.get('dataset_sha256', '')[:10]}`, "
          f"code `{manifest.get('git_commit')}`{' (uncommitted changes)' if manifest.get('git_dirty') else ''}, "
          f"generic fallback {'on' if manifest.get('generic_fallback') else 'off'}.", "",
          "## Headline", "",
          _tbl(["metric", "value"], [
              ["End-to-end latency mean / p50 / p95 (s)", f"{e.get('mean')} / {e.get('p50')} / {e.get('p95')}"],
              ["Tokens per run, in / out (mean)", f"{tok['per_run_in'].get('mean')} / {tok['per_run_out'].get('mean')}"],
              ["Cost per run (USD)", (tok["cost"] or {}).get("per_run_usd", "set MODEL_PRICE_PER_1M in config.py")],
              ["needs_research", f"{o['needs_research_pct']}%"], ["failed", f"{o['failed_pct']}%"],
              ["generic", f"{o['generic_pct']}%"], ["flagged", f"{o['flagged_pct']}%"],
              ["usable draft (drafted+generic+flagged)", f"{o['usable_draft_pct']}%"],
              ["Negative cases behaving as expected", f"{m['expectations']['pass_pct']}% of {m['expectations']['n']}"],
              ["needs_research on non-negative cases", f"{m['non_negative_cases']['needs_research_pct']}% of {m['non_negative_cases']['n']}"],
              ["LLM attempts that were retries", f"{rel['llm_retry_pct']}%"],
              ["Search: backend fallback / empty / failed", f"{rel['search']['fallback_pct']}% / {rel['search']['empty_pct']}% / {rel['search']['failed_pct']}%"],
              ["Embedding calls that fell back to keyword ranking", f"{rel['embedding']['keyword_fallback_pct']}%"],
          ]), "", "## Latency by component (seconds, over runs that ran the component)", ""]
    L.append(_tbl(["component", "runs", "mean", "p50", "p95", "max", "share of e2e"],
                  [[k, s.get("n"), s.get("mean"), s.get("p50"), s.get("p95"), s.get("max"),
                    f"{s['share_of_e2e_pct']}%" if "share_of_e2e_pct" in s else None]
                   for k, s in lat["components_s"].items()]))
    L += ["", f"Sum of component means (all runs) {lat['sum_of_component_means_s']} s vs end-to-end mean {e.get('mean')} s; "
              f"unaccounted (DB writes, thread waits) {lat['unaccounted_mean_s']} s.", "",
          "## Latency per call", ""]
    rows = [[f"LLM {p}", s["n"], s.get("mean"), s.get("p50"), s.get("p95")] for p, s in lat["per_call_s"]["llm"].items()]
    rows += [[f"search {b}", s["n"], s.get("mean"), s.get("p50"), s.get("p95")] for b, s in lat["per_call_s"]["search"].items()]
    rows += [[k, lat["per_call_s"][k]["n"], lat["per_call_s"][k].get("mean"), lat["per_call_s"][k].get("p50"),
              lat["per_call_s"][k].get("p95")] for k in ("fetch", "embedding")]
    L += [_tbl(["call", "n", "mean", "p50", "p95"], rows), "", "## Tokens by component", ""]
    L.append(_tbl(["component", "calls", "in / call", "out / call", "in / run", "out / run", "share of tokens"],
                  [[k, s["calls"], s["in_per_call"], s["out_per_call"], s["in_per_run"], s["out_per_run"],
                    f"{s['share_of_tokens_pct']}%"] for k, s in tok["components"].items()]))
    L += ["", f"Total {tok['total_in']:,} in / {tok['total_out']:,} out. Successful attempts with no token count: "
              f"{tok['ok_attempts_without_token_count']}. Embedding tokens and failed attempts are not counted.", "",
          "## Outcome by category", ""]
    cats = o["by_category"]
    sts = sorted({s for v in cats.values() for s in v})
    L.append(_tbl(["category"] + sts, [[c] + [cats[c].get(s, 0) for s in sts] for c in cats]))
    f_, r_, l_ = m["funnel_mean_per_run"], m["rounds"], m["lookup"]
    L += ["", "## Research funnel (mean per run)", "", _tbl(list(f_), [list(f_.values())]), "",
          f"Rounds used: {r_['distribution']}; {r_['needed_round2_plus_pct']}% needed round 2+; winning round: "
          f"{r_['winning_round']}; search calls per run: {r_['search_calls_per_run']}. "
          f"Lookup found {l_['found_pct']}% of {l_['n']}.", "",
          "## Draft quality (rule-based checks)", ""]
    q = m["quality"]
    L.append(_tbl(["check", "first-attempt pass"], [[k, f"{v['pass_pct']}%"] for k, v in q["first_attempt_checks"].items()]))
    L += ["", f"Redraft {q['redraft_pct']}%, flagged {q['flagged_pct']}%, body words {q['body_words'].get('mean')} "
              f"(max {q['body_words'].get('max')}), unsupported claims per first draft {q['unsupported_claims_per_first_draft']}. "
              f"Hooks: {m['hooks']['type_counts']}.", "",
          "## Limits of this eval", "",
          "- Searches hit the live web, so results drift: a rerun of unchanged code will not give identical numbers. "
          "Run the baseline twice to see the noise before reading a difference as an effect.",
          "- No ground truth for hook relevance, correct person or email quality. Checks here are operational and rule-based.",
          "- With 50 cases, p95 rests on the slowest 2-3 runs; treat it as rough.",
          "- Cases run one at a time, so latencies exclude contention but include any rate limiting by the search backends."]
    return "\n".join(L) + "\n"
