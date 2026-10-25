"""Time a real request end to end and report what the user waits for.

    REWARDSMITH_FAKE_SIM=0 REWARDSMITH_FAKE_LLM=0 python scripts/time_rounds.py "walk backwards" \
        --rounds 2 --coach claude-haiku-5-5 --out walk_coach.json

Runs the worker loop (run_request) on a fresh request in REWARDSMITH_HOME's database
and reads the timings back from it: time to the first score on the board, when round
2's training starts, total time, the final score and success check, and the coach's
calls, latency and cost. --coach off (or a checkout without the coach) is the baseline.
"""

import argparse
import inspect
import json
import re
import statistics
import time
from dataclasses import replace
from pathlib import Path

from rewardsmith import config, db, orchestrator


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("text")
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--coach", default="off", help="a model id, or off")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    config.load_dotenv()
    settings = replace(config.load(), max_rounds=args.rounds, min_rounds=args.rounds, success_score=99)
    conn = db.connect(settings.db_path)
    rid = db.create_request(conn, args.text)
    req = db.claim_next_request(conn)
    assert req["id"] == rid, "another request was queued in this database"

    writer, editor, critic, sim = orchestrator.build(settings)
    extra = {}
    if "coach" in inspect.signature(orchestrator.run_request).parameters:  # absent on the baseline checkout
        extra["coach"] = orchestrator.build_coach(replace(settings, coach=args.coach))
    t0 = time.time()
    best = orchestrator.run_request(
        conn, settings, req, writer, editor, critic, sim, checker=orchestrator.build_checker(settings),
        scener=orchestrator.build_scener(settings), **extra,
    )
    t_end = time.time()
    report = summarize(db.get_request(conn, rid), t0, t_end, best)
    report.update(text=args.text, coach=args.coach, request_id=rid)
    print(json.dumps(report, indent=2))
    if args.out:
        args.out.write_text(json.dumps(report, indent=2))


def summarize(req: dict, t0: float, t_end: float, best) -> dict:
    cands = [c for r in req["rounds"] for c in r["candidates"]]
    scored = sorted(c["status_at"] for c in cands if c["status"] == "scored" and c.get("status_at"))
    stages = req.get("stages") or []

    def stage_start(name: str, n: int) -> float | None:
        starts = [s["started_at"] for s in stages if s["name"] == name and s["round_n"] == n]
        return round(min(starts) - t0, 1) if starts else None

    best_c = next((c for c in cands if best and c["id"] == best.candidate_id), None)
    check = (best_c or {}).get("check_result") or {}
    coach_lines = [e["message"] for e in req["events"] if e["kind"] == "coach"]
    stats = [re.match(r"Coach: (\d+) looks(?:, median ([\d.]+) s)?, (\d+) stopped, \$([\d.]+)", m) for m in coach_lines]
    stats = [m for m in stats if m]
    return {
        "first_score_s": round(scored[0] - t0, 1) if scored else None,
        "round2_writing_s": stage_start("writing", 2),
        "round2_training_s": stage_start("training", 2),
        "total_s": round(t_end - t0, 1),
        "best_score": best.score if best else None,
        "best_passed": best.passed if best else None,
        "best_check": [{k: c.get(k) for k in ("metric", "op", "value", "actual", "pass")} for c in check.get("clauses", [])],
        "scores": [[c["score"] for c in r["candidates"]] for r in req["rounds"]],
        "statuses": [[c["status"] for c in r["candidates"]] for r in req["rounds"]],
        "stages": [{k: (round(s[k] - t0, 1) if s.get(k) else s.get(k)) if k in ("started_at", "ended_at") else s[k]
                    for k in ("name", "round_n", "started_at", "ended_at")} for s in stages],
        "coach_notes": sum(1 for e in req["events"] if e["kind"] == "coach" and e.get("candidate") is not None),
        "coach_calls": sum(int(m[1]) for m in stats),
        "coach_stopped": sum(int(m[3]) for m in stats),
        "coach_median_latency_s": statistics.median(float(m[2]) for m in stats if m[2]) if any(m[2] for m in stats) else None,
        "coach_cost_usd": round(sum(float(m[4]) for m in stats), 4),
        "coach_events": coach_lines,
    }


if __name__ == "__main__":
    main()
