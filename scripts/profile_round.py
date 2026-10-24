"""Time one search round end to end, stage by stage.

    REWARDSMITH_FAKE_SIM=0 REWARDSMITH_FAKE_LLM=0 python scripts/profile_round.py "walk backwards" --out prof.json

Runs the real worker loop (run_request) for --rounds rounds, wrapping the writer,
editor, critic and simulator so every call is timed, and polling each
candidate's progress.json for stage changes (compiling -> training -> rendering
-> done). Prints a table and writes the raw spans to --out.
"""

import argparse
import json
import threading
import time
from dataclasses import replace
from pathlib import Path

from rewardsmith import config, db
from rewardsmith.orchestrator import build, build_ranker, run_request
from rewardsmith.sim.progress import read_progress

T0 = time.monotonic()
SPANS: list[dict] = []
_lock = threading.Lock()


def now() -> float:
    return round(time.monotonic() - T0, 2)


class Timed:
    """Proxy that records a span for every method call on the wrapped object."""

    def __init__(self, name: str, inner):
        self._name, self._inner = name, inner

    def __getattr__(self, attr):
        fn = getattr(self._inner, attr)
        if not callable(fn):
            return fn

        def call(*args, **kwargs):
            start = now()
            try:
                return fn(*args, **kwargs)
            finally:
                with _lock:
                    SPANS.append({"what": f"{self._name}.{attr}", "start": start, "end": now()})

        return call


def poll_stages(artifacts: Path, stop: threading.Event, seen: dict) -> None:
    """Record when each candidate dir first reports each stage."""
    while not stop.is_set():
        for d in artifacts.glob("r*c*"):
            p = read_progress(d)
            if not p:
                continue
            stages = seen.setdefault(d.name, {})
            stages.setdefault(p["stage"], now())
            if p["stage"] == "training" and p["curve"]:
                stages.setdefault("first_eval", now())
            if p["stage"] in ("rendering", "done"):
                stages["last_progress"] = p
        stop.wait(0.25)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("request", nargs="?", default="walk backwards")
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--out", type=Path, default=Path("profile.json"))
    args = ap.parse_args()

    config.load_dotenv()
    settings = replace(config.load(), max_rounds=args.rounds, min_rounds=args.rounds, success_score=99)
    conn = db.connect(settings.db_path)
    writer, editor, critic, sim = build(settings)
    ranker = build_ranker(settings)
    rid = db.create_request(conn, args.request)
    req = db.claim_next_request(conn)

    stop, seen = threading.Event(), {}
    poller = threading.Thread(target=poll_stages, args=(settings.artifacts_dir / rid, stop, seen), daemon=True)
    poller.start()
    start = now()
    best = run_request(
        conn, settings, req, Timed("writer", writer), Timed("editor", editor), Timed("critic", critic), Timed("sim", sim),
        Timed("ranker", ranker) if ranker else None,
    )
    end = now()
    stop.set()
    poller.join()

    detail = db.get_request(conn, rid)
    t_db0 = time.time() - (time.monotonic() - T0)  # wall-clock time of T0
    events = [{"kind": e["kind"], "msg": e["message"][:120], "t": round(e["created_at"] - t_db0, 2)} for e in detail["events"]]
    cands = []
    for r in detail["rounds"]:
        for c in r["candidates"]:
            m = json.loads(c["metrics"]) if isinstance(c["metrics"], str) else (c["metrics"] or {})
            cands.append({
                "round": r["n"], "idx": c["idx"], "status": c["status"], "score": c["score"],
                "episode_reward": m.get("episode_reward"), "forward_velocity": m.get("forward_velocity"),
                "compile_s": m.get("compile_s"), "train_s": m.get("train_s"),
            })
    stages = {k: {s: t for s, t in v.items() if s != "last_progress"} for k, v in seen.items()}
    curves = {k: v.get("last_progress", {}).get("curve") for k, v in seen.items()}
    out = {"request": rid, "total_s": end - start, "best_score": best.score if best else None,
           "spans": SPANS, "events": events, "stages": stages, "candidates": cands, "curves": curves}
    args.out.write_text(json.dumps(out, indent=1))

    print(f"request {rid}: {end - start:.1f} s for {args.rounds} round(s), best {out['best_score']}")
    for s in SPANS:
        print(f"  {s['what']:<16} {s['start']:7.1f} -> {s['end']:7.1f}  ({s['end'] - s['start']:6.1f} s)")
    for name, st in sorted(stages.items()):
        print(f"  {name}: " + "  ".join(f"{k}@{v:.1f}" for k, v in st.items()))
    for c in cands:
        print(f"  r{c['round']}c{c['idx']} {c['status']:<11} score={c['score']} reward={c['episode_reward']} "
              f"fwd={c['forward_velocity']} compile={c['compile_s']} train={c['train_s']}")


if __name__ == "__main__":
    main()
