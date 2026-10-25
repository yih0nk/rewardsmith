"""Run hard prompts end to end (real Claude, real training) and tabulate how each candidate did.

    set -a; . /path/to/.env; set +a
    REWARDSMITH_HOME=/workspace/rs_bench REWARDSMITH_FAKE_SIM=0 REWARDSMITH_FAKE_LLM=0 \
        python scripts/task_bench.py --label after --rounds 1 --lock /workspace/gpu.lock

Each prompt is a fresh request through orchestrator.run_request, with the scene writer and
the checker, exactly as the worker runs it. With --lock, the GPU lock is held per request.
Writes $REWARDSMITH_HOME/bench_<label>.json and prints one row per scored candidate.
"""

import argparse
import fcntl
import json
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ["PYTHONPATH"] = os.pathsep.join(filter(None, [str(ROOT), os.environ.get("PYTHONPATH")]))

from rewardsmith import config, db, orchestrator  # noqa: E402

PROMPTS = (
    "push the ball into the goal zone",
    "walk to the blue box and stop beside it",
    "walk forward on rough terrain",
)
# What to print per candidate, when present.
COLUMNS = (
    "target_reached", "target_final_in", "final_speed", "object_moved", "object_goal_start_dist",
    "object_goal_final_dist", "object_in_goal_final", "object_goal_progress", "forward_velocity", "fell", "target_fell",
)


@contextmanager
def gpu_lock(path: str | None):
    if not path:
        yield
        return
    with open(path, "a") as f:
        t0 = time.time()
        fcntl.flock(f, fcntl.LOCK_EX)
        print(f"  (waited {time.time() - t0:.0f} s for {path})", flush=True)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--lock", help="flock this file around each request (the shared GPU lock)")
    ap.add_argument("prompts", nargs="*", default=list(PROMPTS))
    args = ap.parse_args()

    os.environ["REWARDSMITH_ROUNDS"] = str(args.rounds)
    os.environ["REWARDSMITH_MIN_ROUNDS"] = str(args.rounds)
    settings = config.load()
    conn = db.connect(settings.db_path)
    writer, editor, critic, sim = orchestrator.build(settings)
    checker, scener = orchestrator.build_checker(settings), orchestrator.build_scener(settings)
    out_path = settings.db_path.parent / f"bench_{args.label}.json"
    results = json.loads(out_path.read_text()) if out_path.exists() else []

    for text in args.prompts:
        rid = db.create_request(conn, text)
        req = db.get_request(conn, rid)
        print(f"\n== {rid}: {text}", flush=True)
        t0 = time.time()
        with gpu_lock(args.lock):
            try:
                orchestrator.run_request(conn, settings, req, writer, editor, critic, sim, checker=checker,
                                         scener=scener)
            except orchestrator.Refused:
                db.set_request(conn, rid, status="refused")
        req = db.get_request(conn, rid)
        row = {"label": args.label, "request": rid, "text": text, "wall_s": round(time.time() - t0, 1),
               "status": req.get("status"), "feasibility": req.get("feasibility"),
               "scene": req.get("scene"), "success_spec": req.get("success_spec"),
               "success_code": req.get("success_code"), "success_check_error": req.get("success_check_error"),
               "scene_events": [e["message"] for e in req["events"] if e["kind"] in ("scene", "refusal", "approximate")],
               "candidates": []}
        checked = {c["metric"] for c in req.get("success_spec") or []}
        for cand in _candidates(conn, rid):
            m = cand["metrics"] or {}
            check = cand.get("check_result") or {}
            row["candidates"].append({
                "id": cand["label"], "score": cand["score"], "check": check.get("pass"), "status": cand["status"],
                "critique": (cand.get("critique") or "")[:600],
                "video": cand["video_path"],
                **{k: m[k] for k in m if k in COLUMNS or k in checked or k.startswith("phase_")},
            })
        results.append(row)
        out_path.write_text(json.dumps(results, indent=2))
        _print(row)


def _candidates(conn, rid: str) -> list[dict]:
    req = db.get_request(conn, rid)
    return [{**c, "label": f"r{rnd['n']}c{c['idx']}"} for rnd in req["rounds"] for c in rnd["candidates"]]


def _print(row: dict) -> None:
    verdict = row.get("feasibility") or {}
    if verdict:
        print(f"feasibility: {verdict.get('verdict')}: {verdict.get('reason')} {verdict.get('simplification')} "
              f"{verdict.get('suggestion')}")
    if row.get("status") == "refused":
        print(f"refused in {row['wall_s']} s", flush=True)
        return
    print("spec:", ", ".join(f"{c['metric']} {c['op']} {c['value']:g}" for c in row["success_spec"] or []))
    checked = [c["metric"] for c in row["success_spec"] or [] if c["metric"] not in COLUMNS]
    cols = [c for c in (*checked, *COLUMNS) if any(c in cand for cand in row["candidates"])]
    print("cand   score  check  " + "  ".join(f"{c[:14]:>14}" for c in cols))
    for cand in row["candidates"]:
        vals = "  ".join(f"{cand[c]:>14.3f}" if isinstance(cand.get(c), (int, float)) else f"{'-':>14}" for c in cols)
        score = f"{cand['score']:.1f}" if isinstance(cand["score"], (int, float)) else "-"
        print(f"{cand['id']:<6} {score:>5}  {str(cand['check']):>5}  {vals}")
    print(f"wall {row['wall_s']} s", flush=True)


if __name__ == "__main__":
    main()
