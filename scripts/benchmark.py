"""Benchmark: queue the team plan's behaviors through the real loop, then report the four numbers.

    python scripts/benchmark.py queue                       # every measurable behavior, plus walk_forward
    python scripts/benchmark.py queue --only walk_backwards spin --repeats 2
    python scripts/benchmark.py report --wait               # poll until done, then print and save results

queue adds requests to the same SQLite db the worker reads, each with a fixed success
check from rewardsmith/bench/suite.py, so the checker can't move the target. Run it on
the pod next to a running worker; the runs show up on the board like any other request.
Each queue writes a manifest under <REWARDSMITH_HOME>/benchmark/; report reads the
latest one (or --manifest) and writes .md and .json results next to it.
"""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rewardsmith import config, db  # noqa: E402
from rewardsmith.agents import checker  # noqa: E402
from rewardsmith.bench import report  # noqa: E402
from rewardsmith.bench.suite import BY_SLUG, CASES, Case  # noqa: E402

BASELINE = ROOT / "regression" / "baseline.json"


def queue_case(conn, case: Case) -> str:
    """Create the request and store its success check in one transaction, so the worker
    can't claim it in between and have the checker write a different check."""
    spec = checker.validate([dict(c) for c in case.spec])
    conn.execute("BEGIN IMMEDIATE")
    try:
        rid = db.create_request(conn, case.prompt)
        db.set_success_spec(conn, rid, spec)
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise
    return rid


def cmd_queue(args, settings, conn) -> None:
    unknown = [s for s in args.only if s not in BY_SLUG]
    if unknown:
        sys.exit(f"unknown behavior(s): {', '.join(unknown)}; choose from {', '.join(BY_SLUG)}")
    cases = [BY_SLUG[s] for s in args.only] if args.only else [c for c in CASES if args.reference or not c.reference]
    skipped = [c.slug for c in cases if c.spec is None]
    runs = []
    for case in (c for c in cases if c.spec is not None):
        for _ in range(args.repeats):
            runs.append({"slug": case.slug, "request_id": queue_case(conn, case)})
            print(f"queued {case.slug}: {runs[-1]['request_id']}")
    if skipped:
        print(f"skipped (no metric can check them yet): {', '.join(skipped)}")

    out_dir = settings.db_path.parent / "benchmark"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = out_dir / time.strftime("%Y%m%d-%H%M%S.json")
    manifest.write_text(json.dumps({
        "queued_at": time.time(),
        "settings": {
            "candidates_per_round": settings.candidates_per_round,
            "max_rounds": settings.max_rounds,
            "min_rounds": settings.min_rounds,
            "success_score": settings.success_score,
            "model": settings.model,
        },
        "runs": runs,
    }, indent=2) + "\n")
    print(f"manifest: {manifest}")


def latest_manifest(settings) -> Path:
    found = sorted((settings.db_path.parent / "benchmark").glob("*.json"))
    found = [p for p in found if not p.name.endswith(".results.json")]
    if not found:
        sys.exit("no benchmark manifest yet; run `benchmark.py queue` first")
    return found[-1]


def cmd_report(args, settings, conn) -> None:
    path = args.manifest or latest_manifest(settings)
    manifest = json.loads(path.read_text())
    runs = manifest["runs"]
    while True:
        details = [db.get_request(conn, r["request_id"]) for r in runs]
        missing = [r["request_id"] for r, d in zip(runs, details) if d is None]
        if missing:
            sys.exit(f"requests not in this db: {', '.join(missing)} (wrong REWARDSMITH_DB?)")
        waiting = sum(d["status"] not in report.FINISHED for d in details)
        if not args.wait or not waiting:
            break
        print(f"{waiting} of {len(runs)} still running; checking again in {args.poll:g}s", flush=True)
        time.sleep(args.poll)

    ms = manifest["settings"]
    outcomes = [report.outcome(BY_SLUG[r["slug"]], d, ms["success_score"]) for r, d in zip(runs, details)]
    baseline = json.loads(args.baseline.read_text()) if args.baseline.is_file() else None
    summary = report.summarize(outcomes, ms["max_rounds"], baseline, args.gpu_dollars_per_hour)
    text = report.markdown(summary, outcomes)
    print(text)
    stem = path.with_suffix("")
    stem.with_suffix(".results.md").write_text(text)
    stem.with_suffix(".results.json").write_text(json.dumps({"manifest": manifest, "summary": summary}, indent=2) + "\n")
    print(f"saved {stem.with_suffix('.results.md')}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("queue", help="queue benchmark requests for the worker")
    q.add_argument("--only", nargs="*", default=[], metavar="SLUG", help="just these behaviors")
    q.add_argument("--repeats", type=int, default=1, help="requests per behavior")
    q.add_argument("--no-reference", dest="reference", action="store_false", help="skip the walk_forward comparison")
    r = sub.add_parser("report", help="compute the four numbers from a queued benchmark")
    r.add_argument("--manifest", type=Path, help="default: the latest one")
    r.add_argument("--wait", action="store_true", help="poll until every request has finished")
    r.add_argument("--poll", type=float, default=60.0, help="seconds between polls with --wait")
    r.add_argument("--baseline", type=Path, default=BASELINE, help="regression baseline for the walk_forward comparison")
    r.add_argument("--gpu-dollars-per-hour", type=float, help="to turn GPU minutes into dollars")
    args = ap.parse_args()

    config.load_dotenv()
    settings = config.load()
    conn = db.connect(settings.db_path)
    (cmd_queue if args.cmd == "queue" else cmd_report)(args, settings, conn)


if __name__ == "__main__":
    main()
