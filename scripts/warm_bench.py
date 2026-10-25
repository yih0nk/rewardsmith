"""Before/after for warm starts: real requests end to end (real Claude, real training).

    set -a; . /workspace/rewardsmith/.env; set +a
    REWARDSMITH_HOME=/workspace/rs_engine/bench REWARDSMITH_FAKE_SIM=0 REWARDSMITH_FAKE_LLM=0 \
        python scripts/warm_bench.py --label after --rounds 1 --lock /workspace/gpu.lock
    REWARDSMITH_WARM_START=0 REWARDSMITH_CONTINUE=0 ... --label before     # main's behavior

Each prompt is a fresh request through orchestrator.run_request with the scene writer
and checker, as the worker runs it. Per request it prints the best candidate's critic
score, its task success (the success check's scene clause, else the check), its fall
rate, and each round's wall time; everything goes to $REWARDSMITH_HOME/bench_<label>.json.
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
    "make robot move ball to a target destination",
    "walk to the blue box and stop beside it",
    "stand on your hind legs",
    "walk backwards",
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


def success(metrics: dict, spec: list[dict] | None, check: dict | None) -> float | None:
    """The scene task's success rate if the check has one (e.g. all:reach:ball&push:ball:goal), else pass/fail."""
    for clause in spec or []:
        if ":" in clause["metric"] and isinstance(metrics.get(clause["metric"]), (int, float)):
            return float(metrics[clause["metric"]])
    if isinstance(metrics.get("target_reached"), (int, float)):
        return float(metrics["target_reached"])
    return None if check is None else float(bool(check.get("pass")))


def fell(metrics: dict) -> float | None:
    for key in ("scene_fell", "robust_fell", "fell"):
        if isinstance(metrics.get(key), (int, float)):
            return float(metrics[key])
    return None


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
    rows = json.loads(out_path.read_text()) if out_path.exists() else []

    for text in args.prompts:
        rid = db.create_request(conn, text)
        req = db.get_request(conn, rid)
        print(f"\n== {args.label} {rid}: {text}", flush=True)
        t0 = time.time()
        with gpu_lock(args.lock):
            orchestrator.run_request(conn, settings, req, writer, editor, critic, sim, checker=checker, scener=scener)
        req = db.get_request(conn, rid)
        spec = req.get("success_spec")
        rounds = []
        for rnd in req["rounds"]:
            ends = [s["ended_at"] for s in req["stages"] if s["round_n"] == rnd["n"] and s["ended_at"]]
            cands = []
            for c in rnd["candidates"]:
                m = c["metrics"] or {}
                cands.append({
                    "idx": c["idx"], "score": c["score"], "status": c["status"],
                    "success": success(m, spec, c.get("check_result")), "fell": fell(m),
                    "check": (c.get("check_result") or {}).get("pass"), "video": c["video_path"],
                    **{k: m.get(k) for k in ("train_s", "compile_s", "train_steps", "warm_started", "forward_velocity",
                                             "base_height", "upright")},
                })
            rounds.append({"n": rnd["n"], "wall_s": round(max(ends) - rnd["created_at"], 1) if ends else None,
                           "candidates": cands})
        scored = [c for r in rounds for c in r["candidates"] if isinstance(c["score"], (int, float))]
        best = max(scored, key=lambda c: c["score"], default=None)
        plan = next((e["message"] for e in req["events"] if e["kind"] == "plan"), None)
        row = {"label": args.label, "request": rid, "text": text, "wall_s": round(time.time() - t0, 1),
               "scene": req.get("scene"), "success_spec": spec, "plan": plan, "best": best, "rounds": rounds}
        rows.append(row)
        out_path.write_text(json.dumps(rows, indent=2))
        _print(row)


def _print(row: dict) -> None:
    pct = lambda v: "-" if v is None else f"{100 * v:.1f}%"  # noqa: E731
    print(f"plan: {row['plan']}")
    print("spec:", ", ".join(f"{c['metric']} {c['op']} {c['value']:g}" for c in row["success_spec"] or []))
    for r in row["rounds"]:
        print(f"round {r['n']}: wall {r['wall_s']} s")
        for c in r["candidates"]:
            score = f"{c['score']:.1f}" if isinstance(c["score"], (int, float)) else "-"
            print(f"  c{c['idx']} score {score:>4}  success {pct(c['success']):>6}  fell {pct(c['fell']):>6}  "
                  f"train {c['train_s']} s  steps {c['train_steps']}  {c['status']}")
    b = row["best"]
    if b:
        print(f"BEST {row['label']} | {row['text']} | score {b['score']} | success {pct(b['success'])} | "
              f"fell {pct(b['fell'])} | {b['video']}", flush=True)


if __name__ == "__main__":
    main()
