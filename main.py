"""Streamlit UI. Page 1: New run (input form, live stage timeline, review card).
Page 2: Batch (upload a CSV, run every row in the background, download results).
Page 3: Dashboard (counts, run table, replay of any past run).
Page 4: Health (LLM, search and draft-quality metrics, alerts).

Run:  streamlit run main.py
"""

import io
import json
import time

import pandas as pd
import streamlit as st

import agent
import batch
import config as C
import observability
import store

st.set_page_config(page_title="Outreach pipeline", layout="wide")

ICON = {"started": "⏳", "done": "✅", "warning": "⚠️", "failed": "❌", "skipped": "⏭️"}
STATUS_COLOR = {"running": "blue", "drafted": "green", "generic": "primary", "flagged": "orange",
                "needs_research": "violet", "failed": "red", "approved": "green", "discarded": "gray"}


def badge(status):
    return f":{STATUS_COLOR.get(status, 'gray')}-badge[{status}]"


# =============================================================================
# Timeline (live and replay share this renderer)
# =============================================================================
def render_event(ev, box):
    with box.container(border=True):
        elapsed = f" · {ev['elapsed']}s" if ev.get("elapsed") is not None else ""
        st.markdown(f"{ICON.get(ev['status'], '•')} **{ev['stage']} · {ev['stage_name']}**{elapsed}  \n"
                    f"{ev['summary']}")
        if ev.get("payload"):
            with st.expander("Full output"):
                render_payload(ev["stage"], ev["payload"])


def render_lookup(info):
    st.write("Queries:", [q["query"] for q in info["queries"]])
    if info.get("errors"):
        st.error(info["errors"])
    for c in info.get("candidates", []):
        mark = "★" if c["value"] == info.get("chosen") else "•"
        st.markdown(f"{mark} **{c['value']}** — {len(c['sources'])} source(s), latest {c['latest'] or 'undated'}  \n"
                    f"> {c['evidence'][0]}  \n" + " · ".join(f"[source]({u})" for u in c["sources"]))
    for c in info.get("dropped", []):
        st.markdown(f"🚫 ~~{c['value']}~~  \n*Dropped: {c['drop_reason']}*")
    if not info.get("candidates") and not info.get("dropped"):
        st.write("No candidates found.")
    df = pd.DataFrame(info.get("sources") or [])
    if not df.empty:
        st.dataframe(df.reindex(columns=["title", "url", "query"]), hide_index=True, width="stretch")


def render_payload(stage, payload):
    if stage == 1 and payload.get("field"):
        render_lookup(payload)
    elif stage == 2:
        if payload.get("gaps"):
            st.write("Gaps from earlier rounds:", ", ".join(payload["gaps"]))
        if payload.get("queries"):
            st.dataframe(pd.DataFrame(payload["queries"]), hide_index=True, width="stretch")
        else:
            st.write("No new queries.")
        if payload.get("rejected"):
            st.write("Rejected LLM queries:")
            st.dataframe(pd.DataFrame(payload["rejected"]), hide_index=True, width="stretch")
    elif stage == 3:
        st.write("Results per query:", payload.get("per_query"))
        if payload.get("backends"):
            st.write("Search backend used:", payload["backends"])
        if payload.get("ranking"):
            st.write("Ranking:", payload["ranking"])
        if payload.get("errors"):
            st.error(payload["errors"])
        if payload.get("fetched"):
            st.write("Full-text fetch:", payload["fetched"])
        df = pd.DataFrame((payload.get("sources") or []) + (payload.get("cut") or []))
        if df.empty:
            st.write("No sources gathered.")
        else:
            n_kept = int((df["kept"].fillna(1) == 1).sum()) if "kept" in df else len(df)
            st.caption(f"Kept {n_kept} of {len(df)}; "
                       "sim = best similarity to a seller problem or trigger event.")
            df["full_text"] = df["fulltext"].notna() if "fulltext" in df else False
            st.dataframe(df.reindex(columns=["kept", "sim", "anchor", "full_text", "title", "published_date", "url",
                                             "query", "origin"]),
                         hide_index=True, width="stretch")
    elif stage == 4:
        for s in payload.get("verified", []):
            st.markdown(f"✅ **{s['type']}** ({s['level']}, {s['date'] or 'undated'}) — {s['summary']}  \n"
                        f"> {s['evidence']}  \n" + " · ".join(f"[source]({u})" for u in s["source_urls"]))
        for s in payload.get("dropped", []):
            st.markdown(f"🚫 ~~{s['summary']}~~  \n*Dropped: {s['drop_reason']}*")
        for s in payload.get("merged", []):
            st.markdown(f"🔗 {s['summary']}  \n*New report of an earlier signal; now {len(s['source_urls'])} sources*")
        if not payload.get("verified") and not payload.get("dropped") and not payload.get("merged"):
            st.write("No signals found.")
    elif stage == 5:
        df = pd.DataFrame(payload["ranked"])
        if not df.empty:
            chosen = payload.get("chosen_id")
            df.insert(0, "pick", df["id"].map(lambda i: "★" if i == chosen else
                                              ("runner-up" if i in payload.get("runner_up_ids", []) else "")))
            st.dataframe(df.drop(columns=["id"]), hide_index=True, width="stretch")
        st.caption(f"score = (R/3) × ({C.W_RECENCY}·T + {C.W_PERSONAL}·P); gate R ≥ {C.GATE_MIN_R}")
    elif stage == 6:
        st.text(f"Subject: {payload['subject']}\n\n{payload['body']}")
    elif stage == 7:
        render_checks(payload["checks"])
    else:
        st.json(payload)


def render_checks(checks):
    for c in checks or []:
        icon = "✅" if c["passed"] else ("⚠️" if c["passed"] is None else "❌")
        st.markdown(f"{icon} **{c['check']}** — {c['detail']}")
        if c.get("claims"):
            with st.expander("Grounding claims"):
                for cl in c["claims"]:
                    st.markdown(f"{'✅' if cl['supported'] else '❌'} {cl['claim']}  \n*{cl['support']}*")


class LiveTimeline:
    """Callback for agent events: one slot per stage, updated in place."""

    def __init__(self, parent):
        self.parent = parent
        self.slots = {}

    def __call__(self, ev):
        key = (ev["stage"], ev["stage_name"])
        if key not in self.slots:
            self.slots[key] = self.parent.empty()
        render_event(ev, self.slots[key])


def replay_timeline(run_id):
    for ev in store.get_events(run_id):
        render_event(ev, st.empty())


# =============================================================================
# Review card (Stage 8)
# =============================================================================
def review_card(run_id, live_parent=None):
    run = store.get_run(run_id)
    st.subheader(f"Review · run {run_id}")
    st.markdown(f"**{run['name'] or '(name not found)'}**, {run['role'] or '(role not found)'} "
                f"at **{run['company']}** — status {badge(run['status'])}")
    res = run.get("resolved")
    if res and res.get("value"):
        lookup_notice(res)

    if run["status"] == "needs_research":
        st.info(run["error"] or "No signal passed the relevance gate (R ≥ 2), so no draft was written. "
                                "Add notes or a website and run again.")
        show_signals_table(run_id)
        return
    if run["status"] == "failed":
        st.error(f"Run failed: {run['error']}")
        return
    if run["status"] == "running":
        st.info("This run is still in progress.")
        return
    d = store.current_draft(run_id)
    if d is None:
        st.write("No draft.")
        return

    left, right = st.columns([3, 2])
    with left:
        if d.get("kind") == "generic":
            st.warning("**Generic draft** — research found no personalised hook, so this email speaks to the "
                       "role only (the generic-fallback switch on the Dashboard is on). Add something specific "
                       "before sending, or discard it.")
        if run.get("error") and run["status"] in agent.REVIEWABLE:
            st.warning(run["error"])
        if d["status"] == "flagged":
            st.warning("Draft is flagged: checks failed twice, or a check could not run. "
                       "Review the highlighted items before approving.")
        render_checks(d["check_results"])
        editable = run["status"] in agent.REVIEWABLE
        subject = st.text_input("Subject", agent._pick(d["edited_subject"], d["subject"]), key=f"subj{d['id']}",
                                disabled=not editable)
        body = st.text_area("Body", agent._pick(d["edited_body"], d["body"]), height=320, key=f"body{d['id']}",
                            disabled=not editable)
        if d["edited_body"] is not None or d["edited_subject"] is not None:
            with st.expander("Original (unedited) draft"):
                st.text(f"Subject: {d['subject']}\n\n{d['body']}")

        if editable:
            c1, c2, c3 = st.columns(3)
            if c1.button("Approve", type="primary", key=f"ap{d['id']}"):
                try:
                    agent.approve(run_id, subject, body)
                    st.rerun()
                except ValueError as e:
                    st.error(str(e))
            if c2.button("Save edit", key=f"ed{d['id']}"):
                try:
                    saved = agent.save_edit(run_id, subject, body)
                    st.toast("Edit saved" if saved else "No changes to save")
                    st.rerun()
                except ValueError as e:
                    st.error(str(e))
            with c3.popover("Discard"):
                with st.form(f"discard{d['id']}"):
                    reason = st.text_input("Reason")
                    if st.form_submit_button("Confirm discard"):
                        if reason.strip():
                            agent.discard(run_id, reason.strip())
                            st.rerun()
                        else:
                            st.error("Give a short reason.")

            gated = agent.runner_ups(run_id)
            opts = {s["id"]: f"#{s['rank']} {s['type']} ({s['level']}), score {s['score']:.2f} — {s['summary'][:70]}"
                    for s in gated}
            with st.expander("Regenerate"):
                if run["chosen_signal_id"] is None:  # generic run: no signal to choose from
                    sid = None
                    st.caption("No signal passed the gate; this writes a new generic draft.")
                else:
                    sid = st.selectbox("From signal", list(opts), format_func=opts.get,
                                       index=list(opts).index(d["signal_id"]) if d["signal_id"] in opts else 0,
                                       key=f"rg_sel{d['id']}")
                if st.button("Regenerate draft", key=f"rg{d['id']}"):
                    holder = (live_parent or st).container()
                    with st.spinner("Redrafting... (don't click elsewhere until it finishes)"):
                        status, err = agent.regenerate(run_id, sid, on_event=LiveTimeline(holder))
                    if err:
                        st.error(f"Regenerate failed; the previous draft is kept. {err}")
                    else:
                        st.rerun()
        elif run["status"] == "approved":
            st.success("Approved. Nothing was sent; copy the text below.")
            st.code(agent.final_text(d), language=None)
        elif run["status"] == "discarded":
            acts = [a for a in store.get_review_actions(run_id) if a["action"] == "discard"]
            st.info(f"Discarded: {acts[-1]['reason'] if acts else ''}")

    with right:
        why_this_hook(run_id, d)


def lookup_notice(res):
    n = len(res["sources"])
    msg = (f"The prospect's **{res['field']}** was not given; web search found **{res['value']}** "
           f"({n} source{'s' * (n != 1)}, latest {res['latest'] or 'undated'}). Confirm before sending.  \n"
           f"> {res['evidence']}  \n" + " · ".join(f"[source {i + 1}]({u})" for i, u in enumerate(res["sources"])))
    if res["alternatives"]:
        msg += "  \nOther candidates: " + ", ".join(f"{a['value']} ({len(a['sources'])} source(s))"
                                                    for a in res["alternatives"])
    st.warning(msg)


def why_this_hook(run_id, d):
    sig = store.get_signal(d["signal_id"])
    st.markdown("#### Why this hook")
    if sig:
        found_in = f" · found in research round {sig['round']}" if (sig.get("round") or 1) > 1 else ""
        st.markdown(f"**{sig['type']}** · {sig['level']} · {sig['date'] or 'undated'}{found_in}  \n{sig['summary']}")
        st.markdown(f"> {sig['evidence']}")
        st.markdown(" · ".join(f"[source {i + 1}]({u})" for i, u in enumerate(sig["source_urls"] or [])))
        st.markdown(f"R = **{sig['R']}** ({sig['R_reason']})  \nT = **{sig['T']}** · P = **{sig['P']}** "
                    f"→ score **{sig['score']:.2f}**")
    else:
        st.markdown("No hook: no signal reached R ≥ 2, so this is a generic, role-based draft. "
                    "The signals that were found are below.")
        show_signals_table(run_id)
    others = [s for s in agent.runner_ups(run_id) if s["id"] != d["signal_id"]][:C.N_RUNNER_UPS]
    if others:
        st.markdown("#### Runner-ups")
        for s in others:
            st.markdown(f"- #{s['rank']} **{s['type']}** ({s['level']}, {s['date'] or 'undated'}), "
                        f"score {s['score']:.2f}: {s['summary']}")
    with st.expander("Draft history"):
        for x in store.get_drafts(run_id):
            st.markdown(f"v{x['version']}.{x['attempt']} · {x['status']} · signal {x['signal_id']} · {x['subject']}")
        for a in store.get_review_actions(run_id):
            st.markdown(f"- {a['at']} **{a['action']}** {a['reason'] or ''}")


def show_signals_table(run_id):
    sigs = store.get_signals(run_id)
    if sigs:
        st.dataframe(pd.DataFrame(sigs)[["rank", "type", "level", "summary", "date", "verified", "drop_reason",
                                         "R", "R_reason", "T", "P", "score"]],
                     hide_index=True, width="stretch")


# =============================================================================
# Pages
# =============================================================================
def page_new_run():
    st.title("New run")
    form_col, timeline_col = st.columns([1, 2])
    with form_col:
        with st.form("prospect"):
            name = st.text_input("Prospect name")
            company = st.text_input("Company *")
            role = st.text_input("Role / title")
            website = st.text_input("Company website (optional)")
            notes = st.text_area("Pasted notes (optional)", height=120,
                                 help="e.g. text copied from the prospect's LinkedIn page")
            submitted = st.form_submit_button("Run", type="primary")
        st.caption("Give the prospect name, the role, or both; a missing one is looked up by web search.")
        job = batch_slot()["job"]
        if job is not None and job.running:
            st.warning("A batch is running (Batch page). A single run now competes with it for search "
                       "rate limits.")
        st.caption("Generic fallback: **" + ("on" if agent.generic_fallback_on() else "off")
                   + "** — when no hook is found, " + ("a generic draft is written."
                   if agent.generic_fallback_on() else "the run stops at needs_research.")
                   + " Change it on the Dashboard.")

    with timeline_col:
        st.markdown("#### Stages")
        timeline = st.container()
        if submitted:
            p = {"name": name, "company": company, "role": role, "website": website, "notes": notes}
            missing = agent.validate_input(p)
            if missing:
                form_col.error(f"Required: {', '.join(missing)}")
            else:
                with st.spinner("Running... (clicking elsewhere on the page interrupts the run)"):
                    run_id = agent.run_pipeline(p, on_event=LiveTimeline(timeline))
                st.session_state["current_run"] = run_id
        elif st.session_state.get("current_run"):
            with timeline:
                replay_timeline(st.session_state["current_run"])

    if st.session_state.get("current_run"):
        st.divider()
        review_card(st.session_state["current_run"], live_parent=timeline)


@st.cache_resource
def batch_slot():
    """One slot per server process, shared by every browser tab: the batch keeps
    running (and stays visible) across reruns, page switches and reloads."""
    return {"job": None}


def batch_status_counts(results):
    counts = {}
    for r in results:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    return ", ".join(f"{n} {s}" for s, n in sorted(counts.items())) or "none yet"


def batch_progress(job):
    results = job.snapshot()
    done, total = len(results), len(job.rows)
    st.progress(done / total if total else 1.0, text=f"{done} / {total} rows · {batch_status_counts(results)}")
    cur = job.current
    if cur:
        i, p, t0 = cur
        who = p["name"] or f"({p['role']})"
        st.markdown(f"⏳ Row {i}: **{who}** at **{p['company']}** · {time.time() - t0:.0f}s")
    elif job.running:
        st.markdown("⏳ Starting…")
    if job.running:
        if job.stop_requested:
            st.caption("Stopping after the current row.")
        elif st.button("Stop after current row"):
            job.stop()
            st.rerun(scope="fragment")
    else:
        took = (job.finished or time.time()) - job.started
        msg = f"Batch {'stopped' if job.stop_requested and done < total else 'finished'}: {done} of {total} rows in {took / 60:.1f} min."
        st.success(msg + " Review the drafts on the Dashboard.")
    if results:
        df = pd.DataFrame(results).reindex(columns=batch.OUT_COLS)
        st.dataframe(df, hide_index=True, width="stretch")
        st.download_button("Download results CSV", batch.results_csv(results),
                           file_name=f"{job.label or 'batch'}_results.csv", mime="text/csv")


def page_batch():
    st.title("Batch")
    slot = batch_slot()
    job = slot["job"]

    if job is not None:
        st.markdown(f"#### {'Running' if job.running else 'Last batch'}: {job.label or 'upload'}")
        # Polls while the batch runs; only this block reruns, not the whole page.
        st.fragment(run_every=3 if job.running else None)(batch_progress)(job)
        if job.running:
            st.info("The batch runs on the server: you can switch pages or close this tab. "
                    "Avoid starting single runs meanwhile; parallel searches get rate-limited.")
            return
        st.divider()

    st.markdown("#### New batch")
    st.caption("CSV with a header row: **company** (required), **name**, **role**, **website**, **notes**, in any "
               "order and any case. Each row needs a name, a role, or both. Rows run one after another "
               "(about 1–3 minutes each).")
    up = st.file_uploader("Prospects CSV", type=["csv"])
    if up is None:
        return
    try:
        rows = batch.parse_rows(io.StringIO(up.getvalue().decode("utf-8-sig")))
    except UnicodeDecodeError:
        st.error("Not UTF-8. In Excel, save as 'CSV UTF-8 (Comma delimited)'.")
        return
    except ValueError as e:
        st.error(str(e))
        return
    if not rows:
        st.error("No data rows.")
        return

    bad = {i: m for i, _, m in batch.invalid_rows(rows)}
    preview = pd.DataFrame([{"row": i, **p, "problem": f"missing {', '.join(bad[i])}" if i in bad else ""}
                            for i, p in rows])
    st.dataframe(preview.reindex(columns=["row", "company", "name", "role", "website", "notes", "problem"]),
                 hide_index=True, width="stretch")
    if bad:
        st.error(f"{len(bad)} invalid row(s): each row needs company plus name or role. Fix the file and upload again.")
        return

    start = st.number_input("Start at row", min_value=rows[0][0], max_value=rows[-1][0], value=rows[0][0],
                            help="To resume a stopped batch, start at the first row that did not finish.")
    todo = [(i, p) for i, p in rows if i >= start]
    st.caption(f"{len(todo)} row(s) to run. Generic fallback is **{'on' if agent.generic_fallback_on() else 'off'}**"
               " (change it on the Dashboard).")
    if st.button(f"Run {len(todo)} row(s)", type="primary"):
        slot["job"] = batch.BatchJob(todo, label=up.name.rsplit(".", 1)[0]).start()
        st.rerun()


def generic_switch():
    """On: a run with no usable hook gets a generic, role-based draft. Off: it stops at needs_research.
    Stored in the database, so batch.py follows it too. Applies to runs started after the change."""
    on = st.toggle("Generic draft when no hook is found", value=agent.generic_fallback_on(),
                   help="On: a run whose research finds no signal with R ≥ 2 gets a generic, role-based draft "
                        "(status 'generic'), marked for review. Off: it stops at needs_research. "
                        "Applies to new runs, including batch.py.")
    if on != agent.generic_fallback_on():
        store.set_setting(agent.GENERIC_SETTING, on)
        st.toast(f"Generic fallback {'on' if on else 'off'} for new runs")


def page_dashboard():
    st.title("Dashboard")
    generic_switch()
    runs = store.list_runs()
    if not runs:
        st.info("No runs yet.")
        return
    df = pd.DataFrame(runs)
    c = st.columns(7)
    c[0].metric("Runs", len(df))
    for col, s in zip(c[1:6], ["drafted", "generic", "approved", "flagged", "needs_research"]):
        col.metric(s.replace("_", " ").capitalize(), int((df["status"] == s).sum()))
    dur = df["duration"].dropna()
    c[6].metric("Avg run time", f"{dur.mean():.0f}s" if len(dur) else "–")

    # hook type + score per run
    hook = {}
    for r in runs:
        s = store.get_signal(r["chosen_signal_id"]) if r["chosen_signal_id"] else None
        d = None if s else store.current_draft(r["id"])
        hook[r["id"]] = (s["type"], s["score"]) if s else (("generic", None) if d and d.get("kind") == "generic"
                                                           else (None, None))
    df["hook_type"] = df["id"].map(lambda i: hook[i][0])
    df["score"] = df["id"].map(lambda i: hook[i][1])
    df["looked_up"] = df["resolved"].map(
        lambda r: f"{r['field']}: {'found' if r.get('value') else 'not found'}" if isinstance(r, dict) else "")

    statuses = sorted(df["status"].unique())
    pick = st.multiselect("Filter by status", statuses, default=statuses)
    view = df[df["status"].isin(pick)].reindex(columns=["id", "created_at", "name", "company", "role", "looked_up",
                                                        "status", "rounds", "hook_type", "score", "duration"])
    st.dataframe(view, hide_index=True, width="stretch")

    if view.empty:
        return
    rid = st.selectbox("Open run", view["id"].tolist(),
                       format_func=lambda i: f"#{i} · {df.set_index('id').loc[i, 'name']} "
                                             f"({df.set_index('id').loc[i, 'company']})")
    st.divider()
    tl, rv = st.tabs(["Timeline (replay)", "Draft & review"])
    with tl:
        replay_timeline(rid)
    with rv:
        review_card(rid)


def _table(df, empty="No data yet."):
    if df is None or len(df) == 0:
        st.caption(empty)
    else:
        st.dataframe(df, hide_index=True, width="stretch")


def page_health():
    st.title("Health")
    st.caption("Built from the call records in the database: every LLM attempt, search, page fetch and "
               "embedding call, plus drafts and review actions. Rates are per attempt/call unless noted.")
    rep = observability.report()
    if not rep["n_runs"]:
        st.info("No runs yet.")
        return
    for a in observability.alerts(rep):
        st.warning(a)
    o = rep["overall"]
    c = st.columns(5)
    c[0].metric("Runs", rep["n_runs"])
    c[1].metric("LLM attempts", o.get("llm_attempts", 0))
    c[2].metric("LLM error rate", f"{o['llm_error_pct']}%" if o.get("llm_error_pct") is not None else "–")
    c[3].metric("Search failed", f"{o['search_failed_pct']}%" if o.get("search_failed_pct") is not None else "–")
    t = rep["tokens_per_run"]
    c[4].metric("Tokens / run (median)", f"{t['in']:,} in / {t['out']:,} out" if t else "–")

    llm_t, tool_t, qual_t, run_t = st.tabs(["LLM", "Search & fetch", "Draft quality", "Runs"])
    with llm_t:
        st.subheader("By model")
        _table(rep["llm_by_model"], "No LLM calls recorded yet (runs made before this feature have none).")
        st.subheader("By purpose")
        _table(rep["llm_by_purpose"])
        st.subheader("Errors")
        _table(rep["llm_errors"], "No failed attempts.")
    with tool_t:
        st.subheader("By tool and backend")
        _table(rep["tools"], "No tool calls recorded yet.")
        st.subheader("Errors")
        _table(rep["tool_errors"], "No failed calls.")
        st.subheader("Research funnel (average per run, by round)")
        _table(rep["funnel"])
    with qual_t:
        q = rep["quality"]
        c = st.columns(2)
        c[0].metric("Redraft rate", f"{q['redraft_rate']}%" if q["redraft_rate"] is not None else "–",
                    help="Share of first drafts that failed a check and were redrafted.")
        c[1].metric("Median edit size", f"{q['edit_size']}%" if q["edit_size"] is not None else "–",
                    help="Median share of the body changed by the rep before approving, among edited drafts.")
        st.subheader("First-attempt check results")
        _table(q["checks"])
        st.subheader("Rep actions")
        _table(rep["review"], "No review actions yet.")
    with run_t:
        st.subheader("Status")
        _table(rep["status_mix"])
        st.subheader("Where failed runs stopped")
        _table(rep["failure_stages"], "No failed runs.")
        st.subheader("Stuck runs")
        _table(rep["stuck"][["id", "created_at", "company", "current_stage"]] if len(rep["stuck"]) else None,
               "None.")


PAGES = {"New run": page_new_run, "Batch": page_batch, "Dashboard": page_dashboard, "Health": page_health}
PAGES[st.sidebar.radio("Page", list(PAGES))]()
