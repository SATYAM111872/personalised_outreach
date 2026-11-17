"""SQLite storage for the six records in PLAN.md section 6.
One connection per call so Streamlit's threads never share a connection."""

import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone

from config import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT, name TEXT, company TEXT, role TEXT, website TEXT, notes TEXT,
    status TEXT, current_stage INTEGER, chosen_signal_id INTEGER,
    duration REAL, error TEXT, resolved TEXT, rounds INTEGER, research TEXT
);
CREATE TABLE IF NOT EXISTS stage_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER, stage INTEGER, stage_name TEXT, status TEXT,
    started_at TEXT, ended_at TEXT, elapsed REAL, summary TEXT, payload TEXT
);
CREATE TABLE IF NOT EXISTS sources (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER, url TEXT, title TEXT, snippet TEXT, published_date TEXT,
    query TEXT, origin TEXT, bucket TEXT, sim REAL, anchor TEXT, kept INTEGER,
    round INTEGER, fulltext TEXT
);
CREATE TABLE IF NOT EXISTS signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER, type TEXT, level TEXT, summary TEXT, evidence TEXT,
    source_urls TEXT, date TEXT, verified INTEGER, drop_reason TEXT,
    R INTEGER, R_reason TEXT, T REAL, P REAL, score REAL,
    passed_gate INTEGER, rank INTEGER, round INTEGER
);
CREATE TABLE IF NOT EXISTS drafts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER, signal_id INTEGER, version INTEGER, attempt INTEGER,
    subject TEXT, hook TEXT, bridge TEXT, value TEXT, cta TEXT, body TEXT,
    check_results TEXT, status TEXT, edited_subject TEXT, edited_body TEXT,
    created_at TEXT, kind TEXT
);
CREATE TABLE IF NOT EXISTS review_actions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER, draft_id INTEGER, action TEXT, reason TEXT, at TEXT
);
CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS llm_calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER, stage INTEGER, at TEXT, purpose TEXT, model TEXT, attempt INTEGER,
    outcome TEXT, latency REAL, input_tokens INTEGER, output_tokens INTEGER,
    prompt_chars INTEGER, error TEXT
);
CREATE TABLE IF NOT EXISTS tool_calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER, stage INTEGER, at TEXT, tool TEXT, backend TEXT, outcome TEXT,
    latency REAL, n_results INTEGER, fallbacks INTEGER, detail TEXT
);
CREATE INDEX IF NOT EXISTS idx_llm_calls_run ON llm_calls(run_id);
CREATE INDEX IF NOT EXISTS idx_tool_calls_run ON tool_calls(run_id);
"""

MIGRATIONS = {"runs": {"resolved": "TEXT", "rounds": "INTEGER", "research": "TEXT", "obs": "TEXT"},
              "sources": {"bucket": "TEXT", "sim": "REAL", "anchor": "TEXT", "kept": "INTEGER",
                          "round": "INTEGER", "fulltext": "TEXT"},
              "signals": {"round": "INTEGER"},
              "drafts": {"kind": "TEXT"}}

JSON_COLS = {"payload", "source_urls", "check_results", "resolved", "research", "obs"}


def now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@contextmanager
def _conn():
    os.makedirs(os.path.dirname(DB_PATH) or ".", exist_ok=True)
    con = sqlite3.connect(DB_PATH, timeout=30)
    con.row_factory = sqlite3.Row
    try:
        yield con
        con.commit()
    finally:
        con.close()


def init_db():
    with _conn() as con:
        con.executescript(SCHEMA)
        # Databases created before these features lack the columns.
        for table, cols in MIGRATIONS.items():
            have = {r["name"] for r in con.execute(f"PRAGMA table_info({table})")}
            for col, typ in cols.items():
                if col not in have:
                    con.execute(f"ALTER TABLE {table} ADD COLUMN {col} {typ}")


def _row(r):
    if r is None:
        return None
    d = dict(r)
    for k in JSON_COLS & d.keys():
        if d[k] is not None:
            d[k] = json.loads(d[k])
    return d


def _insert(table, data):
    data = {k: (json.dumps(v, default=str) if k in JSON_COLS else v) for k, v in data.items()}
    cols = ", ".join(data)
    qs = ", ".join("?" for _ in data)
    with _conn() as con:
        cur = con.execute(f"INSERT INTO {table} ({cols}) VALUES ({qs})", list(data.values()))
        return cur.lastrowid


def _update(table, row_id, fields):
    if not fields:
        return
    fields = {k: (json.dumps(v, default=str) if k in JSON_COLS else v) for k, v in fields.items()}
    sets = ", ".join(f"{k} = ?" for k in fields)
    with _conn() as con:
        con.execute(f"UPDATE {table} SET {sets} WHERE id = ?", [*fields.values(), row_id])


def _select(sql, args=()):
    with _conn() as con:
        return [_row(r) for r in con.execute(sql, args).fetchall()]


# --- runs -------------------------------------------------------------------
def create_run(p):
    return _insert("runs", {
        "created_at": now_iso(), "name": p["name"], "company": p["company"], "role": p["role"],
        "website": p.get("website") or "", "notes": p.get("notes") or "",
        "status": "running", "current_stage": 1,
    })


def update_run(run_id, **fields):
    _update("runs", run_id, fields)


def get_run(run_id):
    rows = _select("SELECT * FROM runs WHERE id = ?", (run_id,))
    return rows[0] if rows else None


def list_runs():
    return _select("SELECT * FROM runs ORDER BY id DESC")


# --- events -----------------------------------------------------------------
def add_event(ev):
    return _insert("stage_events", ev)


def get_events(run_id):
    return _select("SELECT * FROM stage_events WHERE run_id = ? ORDER BY id", (run_id,))


# --- sources ----------------------------------------------------------------
def add_sources(run_id, sources):
    for s in sources:
        _insert("sources", {"run_id": run_id, **{k: s.get(k) for k in
                ("url", "title", "snippet", "published_date", "query", "origin", "bucket", "sim", "anchor", "kept",
                 "round", "fulltext")}})


def get_sources(run_id):
    return _select("SELECT * FROM sources WHERE run_id = ? ORDER BY id", (run_id,))


# --- signals ----------------------------------------------------------------
SIGNAL_FIELDS = ("type", "level", "summary", "evidence", "source_urls", "date", "verified",
                 "drop_reason", "R", "R_reason", "T", "P", "score", "passed_gate", "rank", "round")


def add_signal(run_id, sig):
    return _insert("signals", {"run_id": run_id, **{k: sig.get(k) for k in SIGNAL_FIELDS}})


def update_signal(sig_id, **fields):
    _update("signals", sig_id, fields)


def get_signals(run_id):
    return _select("SELECT * FROM signals WHERE run_id = ? ORDER BY rank IS NULL, rank, id", (run_id,))


def get_signal(sig_id):
    rows = _select("SELECT * FROM signals WHERE id = ?", (sig_id,))
    return rows[0] if rows else None


# --- drafts -----------------------------------------------------------------
def add_draft(d):
    return _insert("drafts", {**d, "created_at": now_iso()})


def update_draft(draft_id, **fields):
    _update("drafts", draft_id, fields)


def get_drafts(run_id):
    return _select("SELECT * FROM drafts WHERE run_id = ? ORDER BY id", (run_id,))


def get_draft(draft_id):
    rows = _select("SELECT * FROM drafts WHERE id = ?", (draft_id,))
    return rows[0] if rows else None


def current_draft(run_id):
    """The draft under review: the newest one that is not superseded."""
    rows = _select("SELECT * FROM drafts WHERE run_id = ? AND status != 'superseded' "
                   "ORDER BY id DESC LIMIT 1", (run_id,))
    return rows[0] if rows else None


# --- review actions ---------------------------------------------------------
def add_review_action(run_id, draft_id, action, reason=""):
    return _insert("review_actions", {"run_id": run_id, "draft_id": draft_id,
                                      "action": action, "reason": reason, "at": now_iso()})


def get_review_actions(run_id):
    return _select("SELECT * FROM review_actions WHERE run_id = ? ORDER BY id", (run_id,))


# --- observability (see observability.py) -------------------------------------
LLM_CALL_FIELDS = ("run_id", "stage", "purpose", "model", "attempt", "outcome", "latency",
                   "input_tokens", "output_tokens", "prompt_chars", "error")
TOOL_CALL_FIELDS = ("run_id", "stage", "tool", "backend", "outcome", "latency", "n_results",
                    "fallbacks", "detail")


def add_llm_call(rec):
    return _insert("llm_calls", {"at": now_iso(), **{k: rec.get(k) for k in LLM_CALL_FIELDS}})


def add_tool_call(rec):
    return _insert("tool_calls", {"at": now_iso(), **{k: rec.get(k) for k in TOOL_CALL_FIELDS}})


def get_llm_calls(run_id=None):
    if run_id is None:
        return _select("SELECT * FROM llm_calls ORDER BY id")
    return _select("SELECT * FROM llm_calls WHERE run_id = ? ORDER BY id", (run_id,))


def get_tool_calls(run_id=None):
    if run_id is None:
        return _select("SELECT * FROM tool_calls ORDER BY id")
    return _select("SELECT * FROM tool_calls WHERE run_id = ? ORDER BY id", (run_id,))


def all_drafts():
    return _select("SELECT * FROM drafts ORDER BY id")


def all_review_actions():
    return _select("SELECT * FROM review_actions ORDER BY id")


# --- settings (switches the Dashboard changes; shared with batch.py) ---------
def get_setting(key, default=None):
    rows = _select("SELECT value FROM settings WHERE key = ?", (key,))
    return json.loads(rows[0]["value"]) if rows else default


def set_setting(key, value):
    with _conn() as con:
        con.execute("INSERT INTO settings (key, value) VALUES (?, ?) "
                    "ON CONFLICT(key) DO UPDATE SET value = excluded.value", (key, json.dumps(value)))
