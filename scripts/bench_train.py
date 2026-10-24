"""Time training alone (no Claude): run fixed reward files side by side, like one round does.

    python scripts/bench_train.py examples/rewards/*.py --copies 1 --preset round --out bench.json

Prints, per job, when each progress stage started and the final metrics, so a
settings change can be checked for both speed and quality.
"""

import argparse
import json
import tempfile
import threading
import time
from pathlib import Path

from rewardsmith.sim import parallel
from rewardsmith.sim.progress import read_progress


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("rewards", nargs="+", type=Path)
    ap.add_argument("--copies", type=int, default=1, help="run each reward this many times")
    ap.add_argument("--parallel", type=int, default=4)
    ap.add_argument("--preset", default="round")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    root = Path(tempfile.mkdtemp(prefix="bench_"))
    paths = [r for r in args.rewards for _ in range(args.copies)]
    jobs = [(r.read_text(), root / f"{n}_{r.stem}") for n, r in enumerate(paths)]

    t0 = time.monotonic()
    seen: dict[str, dict[str, float]] = {}
    stop = threading.Event()

    def poll() -> None:
        while not stop.is_set():
            for _, d in jobs:
                p = read_progress(d)
                if p:
                    seen.setdefault(d.name, {}).setdefault(p["stage"], round(time.monotonic() - t0, 1))
            stop.wait(0.25)

    threading.Thread(target=poll, daemon=True).start()
    results = parallel.run_in_processes(jobs, preset=args.preset, max_parallel=args.parallel)
    total = time.monotonic() - t0
    stop.set()

    rows = []
    for (_, d), res in zip(jobs, results):
        row = {"job": d.name, "stages": seen.get(d.name, {})}
        if isinstance(res, str):
            row["error"] = res[-300:]
        else:
            m = res.metrics
            row.update({k: m.get(k) for k in ("episode_reward", "forward_velocity", "lateral_velocity", "yaw_rate",
                                                "base_height", "upright", "fell", "compile_s", "train_s")})
            row["curve"] = res.reward_curve
        rows.append(row)
        print(json.dumps(row))
    print(f"total {total:.1f} s for {len(jobs)} job(s), preset {args.preset}, parallel {args.parallel}")
    if args.out:
        args.out.write_text(json.dumps({"total_s": total, "preset": args.preset, "rows": rows}, indent=1))


if __name__ == "__main__":
    main()
