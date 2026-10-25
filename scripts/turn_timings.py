"""Report what each turn of a conversation made the user wait for, from REWARDSMITH_HOME's db.

    python scripts/turn_timings.py [request ids...]   # default: every request, oldest first

Times count from when the worker started the turn (its "mode" event; else its first stage),
so time spent queued behind other work isn't counted. First video is the first candidate's
rollout.mp4 on disk; first score is the first candidate the critic scored.
"""

import sys
from collections import defaultdict

from rewardsmith import config, db
from rewardsmith.sim import progress


def main() -> None:
    settings = config.load()
    conn = db.connect(settings.db_path)
    ids = sys.argv[1:] or [r["id"] for r in sorted(db.list_requests(conn), key=lambda r: r["created_at"])]
    for rid in ids:
        req = db.get_request(conn, rid)
        mode_events = [e["created_at"] for e in req["events"] if e["kind"] == "mode"]
        start = mode_events[0] if mode_events else min((s["started_at"] for s in req["stages"]), default=None)
        if start is None:
            print(f"{rid}: not started")
            continue
        videos, scores = [], []
        for rnd in req["rounds"]:
            for c in rnd["candidates"]:
                video = progress.candidate_dir(settings.artifacts_dir, rid, rnd["n"], c["idx"]) / "rollout.mp4"
                if video.exists():
                    videos.append(video.stat().st_mtime)
                if c["status"] == "scored" and c["status_at"]:
                    scores.append(c["status_at"])
        stages: dict[str, float] = defaultdict(float)
        for s in req["stages"]:
            stages[s["name"]] += (s["ended_at"] or s["started_at"]) - s["started_at"]
        # Per round: the training processes' own compile and train seconds (slowest candidate),
        # and the rest of the training stage (rollouts, render, robustness, process handoff).
        per_round = []
        for rnd in req["rounds"]:
            runs = [c["metrics"] for c in rnd["candidates"] if c["metrics"] and "compile_s" in c["metrics"]]
            train_stage = [x for x in req["stages"] if x["name"] == "training" and x["round_n"] == rnd["n"]]
            judging = [x for x in req["stages"] if x["name"] == "judging" and x["round_n"] == rnd["n"]]
            writing = [x for x in req["stages"] if x["name"] == "writing" and x["round_n"] == rnd["n"]]
            gate = [x for x in req["stages"] if x["name"] == "compiling" and x["round_n"] == rnd["n"]]
            if not runs or not train_stage:
                continue
            span = lambda xs: sum((x["ended_at"] or x["started_at"]) - x["started_at"] for x in xs)
            compile_s = max(m["compile_s"] for m in runs)
            train_s = max(m["train_s"] for m in runs)
            last_judged = max((x["ended_at"] or 0) for x in judging) if judging else None
            per_round.append(
                f"round {rnd['n']} (K={len(rnd['candidates'])}): writing {span(writing):.0f} s, gate {span(gate):.0f} s, "
                f"PPO compile {compile_s:.0f} s, train {train_s:.0f} s, "
                f"eval+render+handoff {span(train_stage) - compile_s - train_s:.0f} s, "
                f"critic after last video {(last_judged - train_stage[-1]['ended_at']) if last_judged else 0:.0f} s"
            )
        end = max([e["created_at"] for e in req["events"] if e["kind"] in ("done", "error")] or [None]) or None
        fmt = lambda t: f"{t - start:.0f} s" if t else "-"
        print(
            f"{rid} turn {req.get('turn') or 1} {req.get('mode') or 'effort'} {req.get('model') or ''}: "
            f"first video {fmt(min(videos, default=None))}, first score {fmt(min(scores, default=None))}, "
            f"total {fmt(end)}, best {req['best_score']}, rounds {len(req['rounds'])}"
        )
        print("    stages: " + ", ".join(f"{k} {v:.0f} s" for k, v in stages.items()))
        for line in per_round:
            print("    " + line)


if __name__ == "__main__":
    main()
