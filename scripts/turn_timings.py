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
        end = max([e["created_at"] for e in req["events"] if e["kind"] in ("done", "error")] or [None]) or None
        fmt = lambda t: f"{t - start:.0f} s" if t else "-"
        print(
            f"{rid} turn {req.get('turn') or 1} {req.get('mode') or 'effort'} {req.get('model') or ''}: "
            f"first video {fmt(min(videos, default=None))}, first score {fmt(min(scores, default=None))}, "
            f"total {fmt(end)}, best {req['best_score']}, rounds {len(req['rounds'])}"
        )
        print("    " + ", ".join(f"{k} {v:.0f} s" for k, v in stages.items()))


if __name__ == "__main__":
    main()
