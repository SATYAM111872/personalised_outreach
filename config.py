"""Static configuration: seller profile (PLAN.md section 4), query templates,
scoring weights, check rules and timeouts. Edit this file, not agent.py."""

# ---------------------------------------------------------------------------
# Seller profile (assumption A1: one seller, set once).
# Zamp (zamp.ai), from its homepage as read on 2026-11-10.
# The draft may cite anything in proof_points, so each one is copied verbatim
# from zamp.ai. Re-check them before a real campaign.
# ---------------------------------------------------------------------------
SELLER_PROFILE = {
    "product": "Zamp",
    "one_liner": "A digital workforce for finance operations: AI digital employees that run "
                 "ongoing finance jobs end to end (AP approvals, invoice processing and audit, "
                 "purchase requisitions, accounting close, chargebacks) and involve people only when needed.",
    "function": "finance",  # used by the Stage 2 hiring query
    "buyers": ["CFO", "Controller", "Chief Accounting Officer", "VP Finance",
               "Head of Finance Operations", "Head of Accounts Payable", "Head of Procurement",
               "Head of Shared Services", "Head of Payments / Disputes Operations"],
    "problems_solved": [
        "Slow AP approvals and manual invoice processing that do not scale with volume",
        "Purchase requisitions stuck in long approval chains",
        "A month-end accounting close with too many manual touchpoints",
        "Invoice audit that is slow or outsourced to a costly BPO",
        "Chargeback and dispute backlogs that risk missing card-network deadlines",
        "Finance teams that must add headcount every time transaction volume grows",
    ],
    "proof_points": [
        "Reduced Instacart's AP approval times from 2 weeks to 2 days, automating 80% of manual work",
        "Automated 85% of Amgen's purchase requisitions through approval, cutting turnaround time 60%",
        "Eliminated 75% of human touchpoints in Block's accounting close process",
        "Reduced Uber Freight's invoice audit time from 1 day to 1 hour, at 30% lower cost than a BPO",
        "Filed chargebacks in under 10 minutes per case at Wio, with zero missed network deadlines",
        "Processed 95%+ of Mindbody's invoices end-to-end, involving humans only when needed",
    ],
    "allowed_ctas": [
        "Worth a 15-minute call next week?",
        "Want me to send over a one-pager?",
        "Open to a short demo?",
    ],
    "rep_name": "Alex",
    "sign_off": "Best,\nAlex",
    "tone": "Direct, plain, no fluff.",
}

# ---------------------------------------------------------------------------
# Stage 1 - lookup of a missing prospect name or role (one of the two is given)
# ---------------------------------------------------------------------------
NAME_LOOKUP_QUERIES = [            # name missing: who is the {role} of {company}? (web search)
    '"{company}" {role}',
    '{role} of {company} linkedin',
]
NAME_LOOKUP_NEWS_QUERIES = [       # news search: dated, so a recent appointment can beat older coverage
    '"{company}" {role} appointed OR joins OR named',
    '"{company}" new {role} OR succeeds OR replaces OR "steps down"',
]
NAME_LOOKUP_SITE_QUERY = "site:{domain} {role} OR leadership OR team"
ROLE_LOOKUP_QUERIES = [            # role missing: what is {name}'s title at {company}?
    '"{name}" "{company}"',
    '"{name}" {company} title OR role',
    '"{name}" {company} linkedin',
]
ROLE_LOOKUP_NEWS_QUERIES = [
    '"{name}" "{company}" appointed OR promoted OR joins OR named',
]
ROLE_LOOKUP_SITE_QUERY = 'site:{domain} "{name}"'
# A candidate with a dated source this recent beats candidates with more but older
# (or undated) sources: after a change of CFO, older coverage still names the predecessor.
LOOKUP_RECENT_DAYS = 365

# ---------------------------------------------------------------------------
# Stage 2 - query templates
# ---------------------------------------------------------------------------
# Trigger events (indirect hooks) first, then the seller's problems (direct hooks).
COMPANY_QUERIES = [
    '"{company}" news',
    '"{company}" funding OR raises OR valuation',
    '"{company}" acquisition OR acquires OR merger OR expansion',
    '"{company}" revenue OR earnings OR growth',
    '"{company}" CFO OR "chief financial officer" OR controller appointed',
    '"{company}" hiring {function} OR "accounts payable" OR procurement',
    '"{company}" "accounts payable" OR invoices OR "month-end close" OR procurement',
    '"{company}" "finance transformation" OR ERP OR "shared services" OR chargebacks',
]
PERSON_QUERIES = [
    '"{name}" {company}',
    '"{name}" {company} interview OR podcast OR keynote',
    '"{name}" {company} appointed OR joins OR promoted',
    '"{name}" {company} finance OR automation OR AI',
]
WEBSITE_QUERY = "site:{domain} news OR announcement OR blog"

# ---------------------------------------------------------------------------
# Stage 3 - gathering and ranking
# Every result is scored against these anchors plus SELLER_PROFILE
# ["problems_solved"]; a result's score is its best match. Trigger anchors
# describe events that CREATE the seller's problems, so funding or M&A news
# is not ranked low just because it never mentions invoices.
# ---------------------------------------------------------------------------
TRIGGER_ANCHORS = [
    "raised a funding round, new investment or higher valuation",
    "acquired a company or announced a merger",
    "expanding into new countries, markets or business entities",
    "strong revenue growth, earnings results or rising transaction volume",
    "appointed a new CFO, controller or finance leader",
    "hiring accounts payable, procurement or finance operations staff",
    "finance transformation, ERP migration or shared services centre",
    "cost cutting, efficiency programme or restructuring",
    "rising fraud, chargebacks or payment disputes",
    "executive talking about finance automation or AI in finance",
]
RESULTS_PER_QUERY = 5
MAX_SOURCES = 30            # top-scoring results kept for Stage 4 (pasted notes are extra)
MIN_PERSON_SOURCES = 2      # best person-query results always kept, whatever their score
SEARCH_TIMEOUT_S = 20
SEARCH_WORKERS = 4          # parallel queries; higher values get rate-limited
SNIPPET_CHARS = 750         # per source, sent to the LLM

# Full text (fix for thin hooks): fetch the page of the top-ranked results and
# send the relevant paragraphs to Stage 4 instead of the short search snippet.
FETCH_TOP_N = 4             # pages fetched per research round (0 = off)
FETCH_TIMEOUT_S = 10
FETCH_MAX_BYTES = 2_000_000
FULLTEXT_CHARS = 2500       # excerpt per fetched page sent to the LLM
# Login walls, JS redirects and social sites return no article text.
FETCH_SKIP_HOSTS = ("news.google.com", "linkedin.com", "facebook.com", "instagram.com", "x.com",
                    "twitter.com", "youtube.com", "tiktok.com", "reddit.com")

# ---------------------------------------------------------------------------
# Research rounds (fix for "search finds little" and "nothing relevant").
# If no signal passes the gate (R >= GATE_MIN_R) after round 1:
#   round 2 runs the GAP_QUERIES for the gaps found (fixed templates, no LLM),
#   round 3 asks the LLM for new queries, checked by code (see LLM_QUERY_*).
# Each later round searches for NEW results only and adds to the earlier signals.
# ---------------------------------------------------------------------------
MAX_RESEARCH_ROUNDS = 3     # 1 = no retries
FEW_RESULTS = 8             # fewer kept results than this = gap "few_results"
MAX_GAP_QUERIES = 8         # round 2 cap, taken round-robin across the gaps
GAP_QUERIES = {
    # Search found little: drop the strict phrasing, disambiguate with the website.
    "few_results": [
        "{company} company",
        "{company} {domain}",
        '"{company}" about OR "who we are" OR customers',
        "site:{domain} press OR about OR careers OR blog",
    ],
    # No person-level signal: look for the prospect directly.
    "no_person": [
        '"{name}" "{role}"',
        '"{name}" {company} conference OR webinar OR panel OR speaker',
        '"{name}" {company} said OR says OR quoted',
        '"{name}" {company} linkedin',
    ],
    # Signals exist but none links to the seller: aim at finance triggers.
    "no_relevant": [
        '"{company}" "finance team" OR "finance operations" OR "accounting team"',
        '"{company}" careers accounting OR finance OR controller',
        '"{company}" subsidiary OR "new entity" OR "opens office" OR "new market"',
        '"{company}" layoffs OR restructuring OR "cost cutting" OR efficiency',
        '"{company}" "transaction volume" OR customers OR orders OR "record growth"',
    ],
}
LLM_QUERY_MAX = 5           # round 3: queries accepted from the LLM
LLM_QUERY_MAX_CHARS = 150
QUERY_DUP_JACCARD = 0.6     # reject a query this similar to one already run

# ---------------------------------------------------------------------------
# Stage 4 - signals
# ---------------------------------------------------------------------------
SIGNAL_TYPES = ["funding", "hiring", "leadership_change", "product_launch", "expansion",
                "acquisition", "partnership", "exec_statement", "other"]
MERGE_MAX_DAYS = 7          # same event if dates within this many days
MERGE_MIN_JACCARD = 0.35    # ...and summaries overlap this much,
MERGE_ENTITY_TYPES = {"acquisition", "partnership", "funding", "product_launch"}  # ...or share a name

# ---------------------------------------------------------------------------
# Stage 5 - scoring. Starting defaults, not values from a study; tune them
# against the hand-labelled test set (PLAN.md section 7).
# ---------------------------------------------------------------------------
W_RECENCY = 0.6
W_PERSONAL = 0.4
RECENCY_BANDS = [(14, 1.0), (60, 0.7), (180, 0.4)]  # (max_days, T); else UNDATED_T
UNDATED_T = 0.1
P_PERSON = 1.0
P_COMPANY = 0.6
GATE_MIN_R = 2
N_RUNNER_UPS = 2

# ---------------------------------------------------------------------------
# Stage 6/7 - draft and checks
# ---------------------------------------------------------------------------
# Generic fallback: when research ends with no signal passing the gate, write a
# role-based generic draft instead of stopping at needs_research. This is only the
# default; the Dashboard toggle stores the live value in the database.
GENERIC_FALLBACK_DEFAULT = False

BODY_MIN_WORDS = 60
BODY_MAX_WORDS = 120
SUBJECT_MAX_WORDS = 8
MAX_DRAFT_ATTEMPTS = 2      # first draft + one redraft, then flagged

BANNED_PHRASES = [
    "hope this finds you well", "hope this email finds you", "hope you're doing well",
    "i wanted to reach out", "just reaching out", "just checking in", "touch base",
    "circle back", "quick question", "i came across your profile", "synergy",
    "game-changer", "game changer", "revolutionize", "revolutionise", "cutting-edge",
    "best-in-class", "world-class", "leverage", "low-hanging fruit", "no-brainer",
]

HOOK_FRAMING = {
    "funding": "Saw {company} just closed {round}; that usually means pressure to {relevant outcome}...",
    "hiring": "Noticed you're hiring for {role}; that's often a sign {pain} is a priority...",
    "exec_statement": "Read your comments on {topic} in {outlet}: {short paraphrase}. That's the problem we work on...",
    "leadership_change": "Congrats on the move to {role}. The first months usually mean {pressure}...",
    "product_launch": "Saw the {launch}. Scaling that usually means {pain}...",
    "expansion": "Saw the {expansion}. Scaling that usually means {pain}...",
    "acquisition": "With {event}, {integration pain} tends to show up next...",
    "partnership": "With {event}, {integration pain} tends to show up next...",
    "other": "State the event specifically, then connect it to one problem the seller solves.",
}

# ---------------------------------------------------------------------------
# Timeouts / storage
# ---------------------------------------------------------------------------
LLM_TIMEOUT_S = 120              # per call_llm(), all attempts and waits included
LLM_REQUEST_TIMEOUT_S = 40       # one HTTP request to the model; a hung request ends here
LLM_RETRY_BACKOFF_S = [3, 8]     # waits before retrying a 429/503/timeout (attempts = len + 1)
# Used for the retries when the main model is overloaded (503 "high demand") or slow.
# A pinned model, not an alias, from a different capacity pool than the main one.
LLM_FALLBACK_MODEL = "gemini-3.1-flash-lite"  # "-latest" served gemini-3.5-flash-lite on 2026-11-12
DB_PATH = "data/runs.db"

# ---------------------------------------------------------------------------
# Observability (observability.py). Alert thresholds are starting defaults chosen
# without data, not values from a study: set them after looking at a real batch.
# ---------------------------------------------------------------------------
# Eval cost: model name -> (USD per 1M input tokens, USD per 1M output tokens). Empty on purpose:
# fill from the provider's current price page for the exact models in llm.py / LLM_FALLBACK_MODEL.
# Without prices the eval reports tokens only.
MODEL_PRICE_PER_1M = {}

OBS_STUCK_MINUTES = 15           # a run still 'running' after this long is shown as stuck
OBS_ALERTS = {
    "llm_error_pct": 10,          # % of LLM attempts that did not succeed
    "retry_attempt_pct": 15,      # % of LLM attempts that were retries (these go to the fallback model)
    "search_failed_pct": 20,      # % of search calls where every backend failed
    "search_empty_pct": 25,       # % of search calls with no results at all
    "run_failed_pct": 10,
    "needs_research_pct": 30,
    "min_runs_for_rate_alerts": 10,   # run-level rates are noise below this many runs
}
