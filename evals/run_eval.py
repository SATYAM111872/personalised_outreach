"""Eval harness: run the fixed 50-case dataset through the pipeline and record the numbers.

    python evals/run_eval.py run --label baseline          # ~1 h, uses Gemini + web search
    python evals/run_eval.py run --label try2 --resume     # continue an interrupted run
    python evals/run_eval.py run --label smoke --limit 3   # quick check of the harness
    python evals/run_eval.py report baseline               # rebuild metrics/report from the saved DB
    python evals/run_eval.py compare baseline try2         # numbers side by side, with deltas

Each run writes evals/results/<label>/:
    runs.db        isolated database (never touches data/runs.db)
    cases.jsonl    case -> run id, wall time (progress; makes --resume possible)
    manifest.json  what was run: git commit, file hashes, models, config snapshot, dataset hash
    metrics.json   all numbers (nested)        cases.csv  one row per case
    report.md      the readable report

The dataset (evals/dataset_50.csv) is frozen: change it and the hash in the manifest changes,
so compare warns that the runs are not comparable.
"""

import argparse
import csv
import hashlib
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "evals"))

DATASET = ROOT / "evals" / "dataset_50.csv"
RESULTS = ROOT / "evals" / "results"
TRACKED_FILES = ["agent.py", "prompts.py", "config.py", "search.py", "llm.py", "store.py", "observability.py"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_dataset(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    ids = [r["case_id"] for r in rows]
    if len(set(ids)) != len(ids):
        sys.exit("dataset: duplicate case_id")
    return rows


def git(*args):
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:
        return ""


def config_snapshot():
    import config as C
    snap = {}
    for k, v in vars(C).items():
        if k.isupper():
            try:
                json.dumps(v)
                snap[k] = v
            except TypeError:
                snap[k] = repr(v)
    return snap


def build_manifest(args, rows):
    import llm as L
    snap = config_snapshot()
    return {"label": args.label, "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "python": platform.python_version(), "git_commit": git("rev-parse", "--short", "HEAD"),
            "git_dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
            "dataset": str(Path(args.dataset).name), "dataset_sha256": sha(args.dataset), "dataset_rows": len(rows),
            "limit": args.limit, "only": args.only, "generic_fallback": args.generic == "on",
            "models": {"llm": getattr(L.llm, "model", None), "llm_fallback": getattr(L.llm_fallback, "model", None),
                       "embedding": getattr(L.embeddings, "model", None)},
            "file_sha256": {f: sha(ROOT / f) for f in TRACKED_FILES},
            "config_sha256": hashlib.sha256(json.dumps(snap, sort_keys=True).encode()).hexdigest(),
            "config": snap}


def load_cases(out):
    p = out / "cases.jsonl"
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_outputs(out, label, cases, rows):
    import metrics as M
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    order = {r["case_id"]: i for i, r in enumerate(rows)}
    cases = sorted(cases, key=lambda c: order.get(c["case_id"], 1e9))
    m, df = M.compute(str(out / "runs.db"), cases, label)
    m["complete"] = len(cases) == manifest["dataset_rows"] if not (manifest.get("limit") or manifest.get("only")) else None
    (out / "metrics.json").write_text(json.dumps(m, indent=2, default=str), encoding="utf-8")
    df.to_csv(out / "cases.csv", index=False)
    report = M.to_markdown(m, manifest)
    if m["complete"] is False:
        report = report.replace("\n", f"\n\n> PARTIAL: {len(cases)} of {manifest['dataset_rows']} cases.\n", 1)
    (out / "report.md").write_text(report, encoding="utf-8")
    return m


# --- commands ----------------------------------------------------------------------
def cmd_run(args):
    rows = read_dataset(args.dataset)
    if args.only:
        want = set(args.only.split(","))
        rows = [r for r in rows if r["case_id"] in want]
    if args.limit:
        rows = rows[:args.limit]
    if not rows:
        sys.exit("no cases selected")
    out = RESULTS / args.label
    if out.exists() and not args.resume:
        sys.exit(f"{out} exists. Pick another --label, or --resume to continue it.")
    out.mkdir(parents=True, exist_ok=True)

    import agent
    import store
    store.DB_PATH = str(out / "runs.db")  # isolate from data/runs.db
    store.init_db()
    store.set_setting(agent.GENERIC_SETTING, args.generic == "on")

    if not (out / "manifest.json").exists():
        (out / "manifest.json").write_text(json.dumps(build_manifest(args, rows), indent=2), encoding="utf-8")
    done = {c["case_id"] for c in load_cases(out)}
    todo = [r for r in rows if r["case_id"] not in done]
    print(f"{len(rows)} cases, {len(done)} already done, {len(todo)} to run -> {out}", flush=True)

    t_all = time.time()
    try:
        for i, r in enumerate(todo, 1):
            p = {k: r.get(k, "") for k in ("name", "company", "role", "website", "notes")}
            t0, rid, err = time.time(), None, None
            try:
                rid = agent.run_pipeline(p)
            except Exception as e:  # harness-level failure (e.g. invalid row): recorded, not hidden
                err = f"{type(e).__name__}: {e}"
            wall = round(time.time() - t0, 1)
            rec = {"case_id": r["case_id"], "category": r["category"], "expect_status": r.get("expect_status", ""),
                   "run_id": rid, "wall_s": wall, "harness_error": err}
            with open(out / "cases.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
            status = store.get_run(rid)["status"] if rid else "HARNESS ERROR"
            print(f"[{len(done) + i}/{len(rows)}] {r['case_id']} {r['company']}: {status} ({wall}s)", flush=True)
    except KeyboardInterrupt:
        print(f"\nInterrupted. Resume with: python evals/run_eval.py run --label {args.label} --resume", flush=True)
    cases = load_cases(out)
    if cases:
        m = write_outputs(out, args.label, cases, read_dataset(args.dataset))
        print(f"\nWrote {out / 'report.md'}  ({m['n_runs']} runs, {round(time.time() - t_all)}s this session)")


def cmd_report(args):
    out = RESULTS / args.label
    if not (out / "manifest.json").exists():
        sys.exit(f"{out} has no manifest.json")
    write_outputs(out, args.label, load_cases(out), read_dataset(DATASET))
    print(f"Rebuilt {out / 'report.md'}")


def _load(label):
    out = RESULTS / label
    return (json.loads((out / "metrics.json").read_text(encoding="utf-8")),
            json.loads((out / "manifest.json").read_text(encoding="utf-8")), out)


# Skip bookkeeping numbers; compare what moves.
_SKIP = (".n", ".max", "n_cases", "n_runs", "llm_attempts", ".calls", ".runs", "status_counts.", "distribution.",
         "winning_round.", "type_counts.", "by_category.")


def cmd_compare(args):
    import metrics as M
    ma, fa, _ = _load(args.a)
    mb, fb, out_b = _load(args.b)
    A, B = M.flatten(ma), M.flatten(mb)
    notes = []
    if fa["dataset_sha256"] != fb["dataset_sha256"] or fa.get("limit") != fb.get("limit") or fa.get("only") != fb.get("only"):
        notes.append("WARNING: different dataset or case selection. These runs are not comparable.")
    if fa["generic_fallback"] != fb["generic_fallback"]:
        notes.append("WARNING: generic fallback differs between runs; needs_research / generic rates are not comparable.")
    changed = [f for f in TRACKED_FILES if fa["file_sha256"].get(f) != fb["file_sha256"].get(f)]
    cfg = sorted(k for k in set(fa["config"]) | set(fb["config"]) if fa["config"].get(k) != fb["config"].get(k))
    models = {k: (fa["models"].get(k), fb["models"].get(k)) for k in fb["models"] if fa["models"].get(k) != fb["models"].get(k)}
    lines = [f"# {args.a} vs {args.b}", "",
             f"Files changed: {', '.join(changed) or 'none'}. Config keys changed: {', '.join(cfg) or 'none'}. "
             f"Models changed: {models or 'none'}.", ""] + notes + [""]
    rows = []
    for k in sorted(set(A) | set(B)):
        if any(s in k or k.endswith(s) for s in _SKIP):
            continue
        a, b = A.get(k), B.get(k)
        if a is None and b is None:
            continue
        if a == b and not args.all:
            continue
        d = None if a is None or b is None else round(b - a, 3)
        rel = None if d is None or not a else f"{100 * d / abs(a):+.0f}%"
        rows.append([k, a, b, d, rel])
    lines.append(M._tbl(["metric", args.a, args.b, "delta", "rel"], rows) if rows else "No differences.")
    lines += ["", "Live web search makes reruns of identical code differ. Run the same label setup twice to measure that "
                  "noise first; treat deltas inside it as no change."]
    text = "\n".join(lines) + "\n"
    (out_b / f"compare_vs_{args.a}.md").write_text(text, encoding="utf-8")
    print(text)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--label", default=datetime.now().strftime("run_%Y%m%d_%H%M"))
    r.add_argument("--dataset", default=str(DATASET))
    r.add_argument("--limit", type=int, default=0, help="first N cases only (smoke test)")
    r.add_argument("--only", default="", help="comma-separated case_ids")
    r.add_argument("--resume", action="store_true")
    r.add_argument("--generic", choices=["on", "off"], default="off",
                   help="pin the generic-fallback switch for this run (default off, recorded in the manifest)")
    r.set_defaults(fn=cmd_run)
    p = sub.add_parser("report")
    p.add_argument("label")
    p.set_defaults(fn=cmd_report)
    c = sub.add_parser("compare")
    c.add_argument("a", help="baseline label")
    c.add_argument("b", help="new label")
    c.add_argument("--all", action="store_true", help="include unchanged metrics")
    c.set_defaults(fn=cmd_compare)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
