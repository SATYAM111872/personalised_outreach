# Zamp outreach pipeline

From a prospect to a researched, personalised cold-email draft that a person reviews before anything is sent.

**Live demo:** https://zampaidemo.streamlit.app/

Built for the Zamp AI Solutions Associate case study (PS-3). A sales rep enters a company plus a prospect's name and/or role. The pipeline searches the public web, finds a specific, sourced fact to open with (the "hook"), writes a short email about Zamp, checks the draft, and puts it in front of the rep to approve, edit or discard. **Nothing is ever sent automatically.**

---

## What it does

```
Rep input (form or CSV)
   │
   ▼
1  Input check ── name or role missing? ──► look it up (web search + LLM + code checks)
   │
   ▼
┌─ Research round 1, 2 or 3 ───────────────────────────────┐
│ 2  Plan queries     templates → gap templates → LLM-written │
│ 3  Gather sources   search, dedupe, rank, keep 30,          │
│                     fetch full text of the top 4            │
│ 4  Extract signals  LLM, then code checks, merge duplicates │
│ 5  Score            relevance R (LLM) × recency + personal  │
│                     (code); a signal must reach R ≥ 2       │
└──────────────────────────────────────────────────────────┘
   │ no signal passed → next round, or needs_research / generic draft
   ▼
6  Draft (LLM)       subject + short body, one allowed call to action
7  Check             length, banned phrases, names, one ask (code)
                     + grounding of every claim (LLM) → redraft once if needed
8  Human review      approve / edit / discard / regenerate
```

Design rules that run through every stage:

- **Every fact has a source.** Each signal carries the URL it came from. Code checks that the URL was actually gathered and that the source names the company (and the person, for person-level signals).
- **The draft may only use the chosen signal and the seller profile.** The only numbers it may cite are Zamp's published customer results, copied verbatim into `config.py`.
- **Sources are untrusted.** Prompts tell the model to ignore instructions inside search results and pages.
- **Every stage is visible.** Each stage emits an event that the UI shows live and the Dashboard replays later.

The full stage-by-stage behaviour, every query template, every check and every fallback is in [flow.md](flow.md). The original design is in [PLAN.md](PLAN.md), with a diagram in [FLOWCHART.md](FLOWCHART.md).

## Using the app

| Page | What it's for |
|---|---|
| **New run** | One prospect. Company is required, plus a name, a role or both. Website and pasted notes (e.g. text from a LinkedIn profile) are optional. Shows each stage live, then the review card. |
| **Batch** | Upload a CSV and run every row. Runs in a background thread on the server, so you can switch pages or close the tab. Live progress, "Stop after current row", "Start at row" for resuming, and a results CSV to download. |
| **Dashboard** | Every run with its status, hook type and score. Open any run to replay its timeline and review its draft. Also holds the **generic-fallback switch**: when on, a prospect with no usable hook gets a generic, role-based draft (clearly marked) instead of stopping. |
| **Health** | LLM, search and fetch reliability, latency and tokens; research funnel; draft-check pass rates; rep actions; threshold alerts. See [OBSERVABILITY.md](OBSERVABILITY.md). |

**Batch CSV format:** a header row with `company` (required) and any of `name`, `role`, `website`, `notes`, in any order and any case. Each row needs a name, a role or both. For example:

```csv
company,name,role,website,notes
Zepto,Aadit Palicha,,,
Groww,,Chief Financial Officer,groww.in,
```

Rows run one after another, at roughly 1–3 minutes each.

**Run statuses:** `drafted` (ready for review), `generic` (generic draft, no hook found), `flagged` (draft failed its checks twice; review carefully), `needs_research` (nothing usable found; add a role, name, website or notes and rerun), `failed` (an error; the reason is shown), then `approved` or `discarded` after review.

## Running locally

Requires Python 3.10 or 3.11.

```bash
git clone https://github.com/SATYAM111872/ZampAI.git
cd ZampAI
pip install -r requirements.txt
```

Create a `.env` file in the project folder:

```
GEMINI_API_KEY="..."

# Optional: LangSmith tracing of every run
LANGSMITH_TRACING="true"
LANGSMITH_API_KEY="..."
LANGSMITH_PROJECT="zamp_outreach"
```

Then:

```bash
streamlit run main.py                       # the web app
python batch.py prospects.csv               # a batch from the command line
python batch.py prospects.csv --start 7     # resume a stopped batch at row 7
```

The command-line batch writes `<input>_results.csv` after every row. Runs are stored in `data/runs.db` (SQLite), and command-line runs show up on the Dashboard too.

## Project layout

| File | Contents |
|---|---|
| `main.py` | Streamlit UI: New run, Batch, Dashboard |
| `agent.py` | The pipeline: stages 1–7, checks, scoring, review actions |
| `config.py` | Seller profile, query templates, weights, limits and timeouts. **Change behaviour here, not in `agent.py`.** |
| `prompts.py` | Every LLM prompt |
| `search.py` | Web search with fallbacks (DuckDuckGo → Google News RSS → Bing RSS) and safe page fetching |
| `llm.py` | Chat model (`gemini-flash-lite-latest`, temperature 0) and embedding model |
| `store.py` | SQLite storage: runs, stage events, sources, signals, drafts, review actions |
| `batch.py` | CSV batch runner, shared by the command line and the Batch page |
| `tracing.py` | Optional LangSmith tracing; does nothing unless enabled in `.env` |
| `observability.py` | Logs, per-call records (LLM, search, fetch, embedding) and the Health report; see [OBSERVABILITY.md](OBSERVABILITY.md) |

## About the live demo

The demo runs on Streamlit Community Cloud, which affects how it behaves:

- **History resets.** The SQLite database lives on the app's temporary disk, so it's wiped whenever the app restarts, sleeps or is redeployed. Download batch results before leaving.
- **Search is less reliable than locally.** DuckDuckGo limits requests from cloud servers more strictly, so more queries fall back to Google News, which has fewer profile pages.
- **Shared quota.** Every run uses the deployment's Gemini API key.

## Known limitations

Found in test runs (a 15-prospect batch plus repeated single runs). More known gaps are listed in [flow.md](flow.md), section 16.

- **Search reliability is the main weakness.** When DuckDuckGo times out, person lookups fall back to news, which rarely states a mid-level person's title. Some lookups then fail (`needs_research`) and runs take longer.
- **LLM timeouts end the run.** A Gemini call slower than 90 s is not retried, so the run fails at that stage.
- **Lookup picks by number of sources.** Undated, reshared posts can outvote a person's current profile, so a looked-up title can be out of date. Every looked-up value is shown on the review card with its evidence and the alternatives, and must be confirmed before sending.
- **The company check is a whole-word name match.** It doesn't know a company's other names (e.g. Zomato and Eternal) and can't tell a passing mention from a page about the company.