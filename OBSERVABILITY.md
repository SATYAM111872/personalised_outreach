# Observability plan

Goal: for any run, answer *what happened, how long it took, what it cost, what failed and why*, and across runs, *is the pipeline getting better or worse*. Everything here is engineering design for this codebase, not a result from the literature.

## What existed before

| Already there | Where |
|---|---|
| Stage events with timing, summary, payload (live UI + replay) | `stage_events`, `agent.Emitter` |
| Sources (kept and cut, with similarity), signals (with R, T, P, score, drop reason), drafts (with check results), review actions | `store.py` tables |
| Optional LangSmith traces, nested per prospect | `tracing.py` |
| Run status, duration, error, rounds, per-round counts | `runs` |

Gaps: nothing recorded per LLM attempt (model, latency, tokens, retries), per search backend or per fetched host; no logs; no cross-run view; no alerts.

## Layers

1. **Logs.** `logging` logger `outreach`, each line tagged `run=<id> stage=<n>`. Level via env `OUTREACH_LOG_LEVEL`.
2. **Call records.** `llm_calls` (one row per LLM attempt) and `tool_calls` (search, fetch, embedding) in SQLite. Per-run roll-up in `runs.obs`.
3. **Report.** Dashboard → **Health** page: aggregates, error tables, alerts.
4. **Traces.** LangSmith stays as the optional deep-dive (prompts and outputs). Records above are the always-on, local, no-prompt-content layer.

## Observations

Status: ✅ implemented in this change, ◐ partly available from existing tables, ☐ not done.

### A. LLM calls
| # | Observation | Why | Status |
|---|---|---|---|
| A1 | Latency per attempt, p50/p95, by model and by purpose (Lookup, QueryPlan, Extraction, Ratings, DraftParts, Grounding) | Find the slow stage; check `LLM_REQUEST_TIMEOUT_S` / `LLM_TIMEOUT_S` | ✅ |
| A2 | Outcome per attempt: ok / retry / error / deadline / empty | Reliability | ✅ |
| A3 | Retry rate and share of attempts on the fallback model | Primary model overloaded? | ✅ |
| A4 | Error text grouped (503, 429, timeout, parse) | Distinguish capacity from bugs | ✅ |
| A5 | Input/output tokens per attempt, per run | Quota and cost drivers | ✅ (needs the SDK to report `usage_metadata`; see caveats) |
| A6 | Prompt size (chars) | Context growth, truncation risk | ✅ stored, ☐ not charted |
| A7 | Cost in currency | Budgeting | ☐ needs price table; not invented here |
| A8 | Structured-output parse failure rate | Schema/prompt drift | ◐ appears as `error` / `empty` |

### B. Search, fetch, embeddings
| # | Observation | Why | Status |
|---|---|---|---|
| B1 | Backend that answered (ddgs / google-news-rss / bing-rss), calls and latency per backend | README names search as the main weakness | ✅ |
| B2 | Fallback depth (backends skipped before one answered) and their errors | Rate limiting, DNS failures | ✅ |
| B3 | Empty-result rate per query bucket | Query quality | ✅ (bucket in `detail`) |
| B4 | Searches that hit the stage deadline | Hung backends | ✅ |
| B5 | Fetch outcome, latency, characters extracted, per host | Which sites block or give no text | ✅ |
| B6 | Fetch yield: fetched ok / attempted, pages rejected as not mentioning the company | Full-text strategy payoff | ◐ in stage-3 payload `fetched`; funnel shows pages ok |
| B7 | Embedding success, latency, fallback to keyword ranking | Ranking quality silently degrades | ✅ |

### C. Pipeline funnel (per round)
| # | Observation | Status |
|---|---|---|
| C1 | Results gathered → sources kept → pages fetched → signals extracted → dropped (unverified) → passing the R gate | ✅ (Health → Search & fetch) |
| C2 | Which research round produced the winning signal; share of runs needing round 2 or 3 | ◐ `runs.rounds`, `signals.round` |
| C3 | Drop reasons for signals (URL not gathered, person/company not in source, …) | ◐ `signals.drop_reason`, ☐ not aggregated |
| C4 | Gap types that triggered round 2 (few_results / no_person / no_relevant) | ◐ in `runs.research`, ☐ not aggregated |
| C5 | Lookup (name/role) success rate, candidates per lookup | ◐ `runs.resolved` |

### D. Output quality
| # | Observation | Status |
|---|---|---|
| D1 | First-attempt pass rate per check (length, banned phrases, names, one ask, grounding), by draft kind | ✅ |
| D2 | Redraft rate (first draft failed a check) | ✅ |
| D3 | Grounding checks that could not run (`passed = null`) | ✅ shown as "could not run" |
| D4 | Rep actions: approve / edit / discard / regenerate, regenerate failures | ✅ |
| D5 | Edit size: share of the body the rep changed before approving | ✅ (1 − difflib similarity) |
| D6 | Discard reasons | ◐ `review_actions.reason`, ☐ not aggregated |
| D7 | Hook type mix and score distribution | ◐ Dashboard table |
| D8 | Outcome (reply, meeting) | ☐ not captured; the only real quality signal, needs a feedback field |

### E. Run health
| # | Observation | Status |
|---|---|---|
| E1 | Status mix (drafted, generic, flagged, needs_research, failed, approved, discarded) | ✅ |
| E2 | Stage where failed runs stopped | ✅ |
| E3 | Runs stuck in `running` for over `OBS_STUCK_MINUTES` | ✅ |
| E4 | Run duration; time per stage | ◐ `runs.duration`, `stage_events.elapsed` |
| E5 | Batch throughput and ETA | ◐ existing batch progress |

### F. Alerts (Health page banner)
Breach messages for: LLM attempt error rate, retry-attempt share, search all-backends-failed rate, search empty rate, failed-run rate, needs_research rate, stuck runs. Thresholds are in `config.OBS_ALERTS`. They are **starting guesses chosen without data**; set them after looking at a real batch. Run-level rates are only checked after `min_runs_for_rate_alerts` runs.

## Not done, and next steps
- **D8 outcome tracking** is the observation that matters most for research claims; everything above measures process health, not whether emails work.
- Aggregate C3, C4, D6 on the Health page.
- Price table for A7.
- Exporting records (CSV / OpenTelemetry) if a lab-wide dashboard is wanted.
- A held-out, hand-labelled set (PLAN.md section 7) to track hook relevance (R) accuracy over time; LLM-rated R is not validated by these metrics.

## Caveats
- Token counts are collected through a LangChain callback reading `usage_metadata`. I could not run the live Gemini path (dependencies not installed in the environment I tested in), so confirm tokens are non-zero after one real run. If they are 0 or null, the model/SDK combination does not report usage that way.
- Tested with stubbed models: retry → fallback recording, error recording, search and fetch wrappers, context propagation into worker threads, run summary, report and alerts. Not tested: the Streamlit Health page rendering, and anything against the real APIs.
- Records live in `data/runs.db`, which is wiped on Streamlit Community Cloud restarts (see README), so the Health page there only covers the current session.
- Per-attempt rates, not per-call: a call that retried once counts as two attempts.
- Records contain error text and hostnames, not prompts or page text. Logs contain company, name and role, so treat them as personal data like the rest of the run data.
