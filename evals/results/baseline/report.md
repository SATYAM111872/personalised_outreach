# Eval report: baseline

50 cases, 50 produced a run. Dataset `ee8d13b137`, code `1550b10` (uncommitted changes), generic fallback off.

## Headline

| metric | value |
|---|---|
| End-to-end latency mean / p50 / p95 (s) | 62.93 / 62.2 / 125.89 |
| Tokens per run, in / out (mean) | 9257.0 / 1648.0 |
| Cost per run (USD) | set MODEL_PRICE_PER_1M in config.py |
| needs_research | 32.0% |
| failed | 0.0% |
| generic | 0.0% |
| flagged | 0.0% |
| usable draft (drafted+generic+flagged) | 68.0% |
| Negative cases behaving as expected | 100.0% of 5 |
| needs_research on non-negative cases | 24.4% of 45 |
| LLM attempts that were retries | 15.4% |
| Search: backend fallback / empty / failed | 59.0% / 0.0% / 0.3% |
| Embedding calls that fell back to keyword ranking | 56.1% |

## Latency by component (seconds, over runs that ran the component)

| component | runs | mean | p50 | p95 | max | share of e2e |
|---|---|---|---|---|---|---|
| Input | 50 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0% |
| Find prospect | 46 | 17.29 | 11.87 | 33.72 | 34.57 | 25.3% |
| Plan queries | 35 | 0.1 | 0.03 | 0.05 | 2.54 | 0.1% |
| Gather sources | 35 | 48.96 | 44.1 | 83.72 | 88.43 | 54.5% |
| Extract signals | 35 | 8.48 | 7.89 | 18.65 | 23.09 | 9.4% |
| Score & pick hook | 35 | 3.35 | 2.39 | 7.33 | 8.25 | 3.7% |
| Draft | 34 | 3.08 | 1.79 | 7.91 | 13.4 | 3.3% |
| Check draft | 34 | 3.04 | 1.81 | 7.85 | 9.53 | 3.3% |

Sum of component means (all runs) 62.69 s vs end-to-end mean 62.93 s; unaccounted (DB writes, thread waits) 0.25 s.

## Latency per call

| call | n | mean | p50 | p95 |
|---|---|---|---|---|
| LLM DraftParts | 35 | 2.07 | 1.66 | 3.62 |
| LLM Extraction | 41 | 5.18 | 4.22 | 9.71 |
| LLM Grounding | 35 | 2.18 | 1.69 | 4.52 |
| LLM Lookup | 46 | 2.72 | 2.45 | 5.89 |
| LLM QueryPlan | 1 | 2.39 | 2.39 | 2.39 |
| LLM Ratings | 35 | 2.28 | 2.07 | 4.15 |
| search bing-rss | 22 | 11.88 | 8.28 | 33.01 |
| search ddgs-news | 79 | 4.55 | 1.4 | 20.94 |
| search ddgs-web | 304 | 7.58 | 4.92 | 25.71 |
| search google-news-rss | 226 | 14.78 | 8.76 | 46.21 |
| fetch | 129 | 1.32 | 1.02 | 2.68 |
| embedding | 18 | 3.13 | 2.98 | 4.19 |

## Tokens by component

| component | calls | in / call | out / call | in / run | out / run | share of tokens |
|---|---|---|---|---|---|---|
| Find prospect | 46 | 3151.0 | 329.0 | 3151.0 | 329.0 | 29.4% |
| Plan queries | 1 | 902.0 | 299.0 | 902.0 | 299.0 | 0.2% |
| Extract signals | 41 | 5912.0 | 1171.0 | 6925.0 | 1371.0 | 53.3% |
| Score & pick hook | 35 | 736.0 | 201.0 | 757.0 | 207.0 | 6.0% |
| Draft | 35 | 737.0 | 147.0 | 759.0 | 151.0 | 5.7% |
| Check draft | 35 | 660.0 | 195.0 | 679.0 | 201.0 | 5.5% |

Total 462,864 in / 82,404 out. Successful attempts with no token count: 0. Embedding tokens and failed attempts are not counted.

## Outcome by category

| category | drafted | needs_research |
|---|---|---|
| both | 4 | 0 |
| name_only | 8 | 0 |
| negative | 0 | 5 |
| notes | 1 | 2 |
| role_only | 21 | 9 |

## Research funnel (mean per run)

| results | kept | pages_ok | signals | dropped | passing |
|---|---|---|---|---|---|
| 65.1 | 34.7 | 1.1 | 3.8 | 0.5 | 2.5 |

Rounds used: {1: 30, 2: 4, 3: 1}; 14.3% needed round 2+; winning round: {1: 30, 2: 4}; search calls per run: 12.7. Lookup found 67.4% of 46.

## Draft quality (rule-based checks)

| check | first-attempt pass |
|---|---|
| length | 100.0% |
| banned_phrases | 100.0% |
| names | 100.0% |
| one_ask | 100.0% |
| grounding | 97.1% |

Redraft 2.9%, flagged 0.0%, body words 88.0 (max 114.0), unsupported claims per first draft 0.03. Hooks: {'leadership_change': 10, 'acquisition': 8, 'other': 6, 'expansion': 5, 'funding': 2, 'product_launch': 1, 'hiring': 1, 'partnership': 1}.

## Limits of this eval

- Searches hit the live web, so results drift: a rerun of unchanged code will not give identical numbers. Run the baseline twice to see the noise before reading a difference as an effect.
- No ground truth for hook relevance, correct person or email quality. Checks here are operational and rule-based.
- With 50 cases, p95 rests on the slowest 2-3 runs; treat it as rough.
- Cases run one at a time, so latencies exclude contention but include any rate limiting by the search backends.
