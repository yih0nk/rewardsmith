"""Quality regression suite: train fixed rewards and check the robot still does the job.

    python scripts/regression.py --record    # measure and write regression/baseline.json
    python scripts/regression.py --compare   # measure and compare against it; exit 1 on FAIL

Each case trains one hand-written reward from examples/rewards/ through the same
path the worker uses (PlaygroundSimulator.run_many), one case at a time so the
timings don't contend. Training and the rollout both use seed 0, the production
default. A case FAILs if its behavior check fails, or if its final
episode_reward is more than 5% below the baseline. Timings are reported, never
failed: they are for speed branches to quote.

Run it on the GPU pod under the shared lock (see the README).
"""

import argparse
import json
import operator
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
# Import this checkout's rewardsmith, here and in the training subprocesses, even when
# the venv has another checkout installed in editable mode (as on the shared pod).
sys.path.insert(0, str(ROOT))
os.environ["PYTHONPATH"] = os.pathsep.join(filter(None, [str(ROOT), os.environ.get("PYTHONPATH")]))
BASELINE = ROOT / "regression" / "baseline.json"
REWARD_DROP = 0.05  # FAIL if episode_reward falls more than this fraction below baseline
OPS = {"<": operator.lt, ">": operator.gt}
KEPT_METRICS = ("forward_velocity", "lateral_velocity", "yaw_rate", "base_height", "upright", "fell")


@dataclass(frozen=True)
class Case:
    name: str
    reward: str  # path relative to the repo root
    checks: tuple[tuple[str, str, float], ...]  # (metric, op, value), all must hold


NO_FALL = ("fell", "<", 0.5)
CASES: tuple[Case, ...] = (
    Case("walk_backwards", "examples/rewards/walk_backwards.py", (("forward_velocity", "<", -0.3), NO_FALL)),
    Case("walk_forward", "examples/rewards/walk_forward.py", (("forward_velocity", ">", 0.3), NO_FALL)),
    Case("spin_in_place", "examples/rewards/spin_in_place.py",
         (("yaw_rate", ">", 1.0), ("forward_velocity", "<", 0.3), ("forward_velocity", ">", -0.3), NO_FALL)),
)


def check(metrics: dict[str, Any], checks: tuple[tuple[str, str, float], ...]) -> list[dict[str, Any]]:
    out = []
    for metric, op, value in checks:
        actual = metrics.get(metric)
        ok = actual is not None and OPS[op](float(actual), value)
        out.append({"metric": metric, "op": op, "value": value, "actual": actual, "pass": ok})
    return out


def run_case(sim, case: Case, out_root: Path) -> dict[str, Any]:
    out_dir = out_root / case.name
    start = time.perf_counter()
    [result] = sim.run_many([((ROOT / case.reward).read_text(), out_dir)])
    wall_s = round(time.perf_counter() - start, 1)
    if isinstance(result, str):
        return {"error": result[-2000:], "wall_s": wall_s, "behavior_pass": False}
    m = result.metrics
    clauses = check(m, case.checks)
    return {
        "episode_reward": m.get("episode_reward"),
        "behavior_pass": all(c["pass"] for c in clauses),
        "checks": clauses,
        "compile_s": m.get("compile_s"),
        "train_s": m.get("train_s"),
        "wall_s": wall_s,
        "metrics": {k: m[k] for k in KEPT_METRICS if k in m},
    }


def compare(run: dict[str, Any], baseline: dict[str, Any]) -> list[dict[str, Any]]:
    """One row per baseline case: status PASS or FAIL, and why."""
    rows = []
    for name, base in baseline["cases"].items():
        cur = run["cases"].get(name)
        reasons = []
        if cur is None:
            reasons.append("not run")
        elif "error" in cur:
            reasons.append(f"error: {cur['error'].splitlines()[-1] if cur['error'] else '?'}")
        else:
            reasons += [f"check {c['metric']} {c['op']} {c['value']} failed (got {c['actual']})"
                        for c in cur["checks"] if not c["pass"]]
            floor = base["episode_reward"] - REWARD_DROP * abs(base["episode_reward"])
            if cur["episode_reward"] is None or cur["episode_reward"] < floor:
                reasons.append(f"episode_reward {cur['episode_reward']} < {floor:.3f} (baseline - 5%)")
        rows.append({"case": name, "status": "FAIL" if reasons else "PASS", "reasons": reasons,
                     "current": cur or {}, "baseline": base})
    return rows


def format_rows(rows: list[dict[str, Any]]) -> str:
    def delta(cur, base, key):
        a, b = cur.get(key), base.get(key)
        if a is None or b is None:
            return f"{key} {a}"
        return f"{key} {a} ({a - b:+.1f})"

    lines = []
    for r in rows:
        cur, base = r["current"], r["baseline"]
        lines.append(f"{r['status']}  {r['case']:<16} " + "  ".join(
            delta(cur, base, k) for k in ("episode_reward", "compile_s", "train_s", "wall_s")))
        lines += [f"      - {why}" for why in r["reasons"]]
    return "\n".join(lines)


def run_suite(sim, cases: tuple[Case, ...], out_root: Path, preset: str, impl: str | None) -> dict[str, Any]:
    start = time.perf_counter()
    results = {}
    for case in cases:
        print(f"training {case.name} ({case.reward}) ...", flush=True)
        results[case.name] = run_case(sim, case, out_root)
        print(f"  {json.dumps(results[case.name])}", flush=True)
    return {
        "preset": preset,
        "impl": impl,
        "git_sha": _git_sha(),
        "gpu": _gpu_name(),
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "total_wall_s": round(time.perf_counter() - start, 1),
        "reward_drop_limit": REWARD_DROP,
        "cases": results,
    }


def _git_sha() -> str | None:
    """HEAD of this checkout, or REWARDSMITH_GIT_SHA for a copy with no .git (as on the pod)."""
    if os.environ.get("REWARDSMITH_GIT_SHA"):
        return os.environ["REWARDSMITH_GIT_SHA"]
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _gpu_name() -> str | None:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                             capture_output=True, text=True, check=True).stdout
        return out.splitlines()[0].strip() or None
    except (OSError, subprocess.CalledProcessError, IndexError):
        return None


def main(argv: list[str] | None = None, sim=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--record", action="store_true", help="measure and write the baseline")
    mode.add_argument("--compare", action="store_true", help="measure and compare against the baseline")
    ap.add_argument("--preset", help="training preset (default: round to record, the baseline's to compare)")
    ap.add_argument("--impl", help="jax or warp (default: warp on a CUDA GPU)")
    ap.add_argument("--only", nargs="+", metavar="CASE", help="run only these cases")
    ap.add_argument("--baseline", type=Path, default=BASELINE)
    ap.add_argument("--out", type=Path, default=ROOT / ".rewardsmith" / "regression",
                    help="where videos, logs and last_run.json go")
    args = ap.parse_args(argv)

    baseline = json.loads(args.baseline.read_text()) if args.compare else None
    preset = args.preset or (baseline["preset"] if baseline else "round")
    if baseline and preset != baseline["preset"]:
        ap.error(f"the baseline was recorded with --preset {baseline['preset']}; rewards from {preset} don't compare")
    cases = tuple(c for c in CASES if not args.only or c.name in args.only)

    if sim is None:
        from rewardsmith.sim.playground import PlaygroundSimulator

        sim = PlaygroundSimulator(preset=preset, impl=args.impl)
    run = run_suite(sim, cases, args.out, preset, getattr(sim, "impl", args.impl))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "last_run.json").write_text(json.dumps(run, indent=2))

    if args.record:
        failed = [name for name, r in run["cases"].items() if not r["behavior_pass"]]
        if failed:
            print(f"not recording: behavior check failed for {', '.join(failed)}", file=sys.stderr)
            return 1
        args.baseline.parent.mkdir(parents=True, exist_ok=True)
        args.baseline.write_text(json.dumps(run, indent=2) + "\n")
        print(f"baseline written to {args.baseline} (total {run['total_wall_s']}s)")
        return 0

    if args.only:
        baseline = {**baseline, "cases": {k: v for k, v in baseline["cases"].items() if k in args.only}}
    rows = compare(run, baseline)
    print(f"\nbaseline {baseline.get('git_sha')} vs now {run['git_sha']}, preset {preset}, total {run['total_wall_s']}s")
    print(format_rows(rows))
    ok = all(r["status"] == "PASS" for r in rows)
    print("\nREGRESSION PASS" if ok else "\nREGRESSION FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
