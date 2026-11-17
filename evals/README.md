# Evals

A fixed 50-case dataset run through the whole pipeline, giving the same set of numbers every time, so a change to prompts, config, models or code can be compared against a baseline.

```powershell
python evals/run_eval.py run --label baseline          # about 1 hour; uses Gemini and live web search
python evals/run_eval.py run --label baseline --resume # continue after Ctrl-C or a crash
python evals/run_eval.py run --label smoke --limit 3   # check the harness on 3 cases
python evals/run_eval.py report baseline               # rebuild report.md / metrics.json from the saved DB
python evals/run_eval.py compare baseline my_change    # side-by-side numbers with deltas
```

Workflow: run `baseline` once on the current code, change something, run again under a new label, `compare`. Run nothing else against Gemini or the search backends at the same time: it distorts latency and triggers rate limits.

## Dataset (`dataset_50.csv`, frozen)

| category | cases | purpose |
|---|---|---|
| name_only | 8 | Stage 1 must look up the role |
| both | 4 | no lookup |
| role_only | 30 | Stage 1 must look up the name (mix of Indian and global companies, several are Zamp's published customers) |
| notes | 3 | pasted-notes path |
| negative | 5 | fictional companies; `expect_status` = `needs_research` |

Editing the file changes its hash in `manifest.json`, and `compare` then warns that runs are not comparable. Add a new dataset file instead of editing this one.

## What each run saves (`evals/results/<label>/`)

- `report.md`: the readable report. `metrics.json`: every number. `cases.csv`: one row per case (status, per-component latency, tokens, funnel counts, check results).
- `manifest.json`: git commit and whether the tree was dirty, SHA-256 of agent/prompts/config/search/llm/store/observability, model names, a full snapshot of `config.py` constants, dataset hash, whether generic fallback was on. `compare` lists which files, config keys and models differ.
- `runs.db` (git-ignored) and `cases.jsonl` (progress).

The eval uses its own database and pins the generic-fallback switch (`--generic on|off`, default off), so neither `data/runs.db` nor the Dashboard setting affects it.

## Numbers produced

**Latency.** End-to-end (mean, p50, p95, max). Per component: Input, Find prospect, Plan queries, Gather sources, Extract signals, Score and pick hook, Draft, Check draft, each with the number of runs that executed it and its share of end-to-end time. Sum of components against end-to-end, with the unaccounted remainder. Per call: each LLM purpose, each search backend, fetch, embedding.

**Tokens.** Per run (in/out). Per component: calls, tokens per call, tokens per run, share of all tokens. Cost per run if prices are set in `config.MODEL_PRICE_PER_1M` (empty on purpose; fill from the provider's current price page).

**Outcome.** Status mix, `needs_research` / `failed` / `generic` / `flagged` rates, usable-draft rate, outcome by category, negative cases behaving as expected, `needs_research` on non-negative cases.

**Reliability.** LLM attempts: ok, retry, error, deadline, empty. Search: backend fallback rate, empty, failed, backend mix. Fetch ok rate. Embedding calls that fell back to keyword ranking. Share of runs with any tool failure.

**Pipeline behaviour.** Research funnel (results, kept, pages fetched, signals, dropped, passing). Rounds used and the round that produced the winning hook. Lookup success rate. Hook types and scores.

**Draft checks (rule-based).** First-attempt pass rate per check, redraft rate, body length, unsupported claims per first draft.

## Further observations worth adding (not implemented)

1. **Labelled correctness.** Hand-label, for each case, the right person or role, and whether the chosen hook is usable. Gives lookup accuracy and hook precision; nothing here measures those today. This is the biggest gap.
2. **Noise estimate.** Run the baseline 2-3 times; the spread between identical-code runs is the smallest difference you can trust.
3. **Cached-search mode.** Record search and fetch results once and replay them, so a prompt or model change is compared on identical inputs and latency noise from the web disappears. Needs a recording layer in `search.py`.
4. **Per-source usefulness.** Which backends and hosts produce the sources behind chosen hooks (join `sources`, `signals`, `tool_calls`).
5. **Retries by stage.** Which component absorbs the transient LLM errors.
6. **LLM-judge scores** for draft quality or grounding, with a stated rubric. Must itself be validated against a few human ratings before use.
7. **Embedding tokens** (not captured now) and cost of failed attempts.
8. **Time budgets.** Share of runs over a latency target you choose.

## Limits

- Live web search changes between runs, so identical code gives slightly different numbers.
- Without ground truth (item 1 above), the eval shows whether the pipeline got faster, cheaper or more reliable, and whether rule-based checks still pass. It does not show that the emails are better.
- With 50 cases, p95 depends on the 2-3 slowest runs.
- Token counts rely on the SDK reporting usage; check `ok_attempts_without_token_count` in the report.
