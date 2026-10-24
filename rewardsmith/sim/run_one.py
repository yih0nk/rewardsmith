"""Train one reward in its own process. parallel.py starts one of these per candidate.

    python -m rewardsmith.sim.run_one <reward.py> <out_dir> [--preset round] [--impl jax] [--scene scene.json] [--fake]
    python -m rewardsmith.sim.run_one --serve <job.json> [--fake]

Writes <out_dir>/result.json, plus frames/NN.png for the critic. On failure,
result.json holds {"error": "..."} and the exit code is 1.

--serve starts a warm process: it does the slow imports first, then waits for
parallel.py to write job.json ({"reward", "out_dir", "preset", "impl"}) and runs
that job. The imports take ~15 s on the pod, so they overlap the writer's call.
"""

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

RESULT_FILE = "result.json"
FRAMES_DIR = "frames"


def write_result(out_dir: Path, result) -> None:
    frames_dir = out_dir / FRAMES_DIR
    frames_dir.mkdir(parents=True, exist_ok=True)
    names = []
    for i, png in enumerate(result.frames):
        name = f"{i:02d}.png"
        (frames_dir / name).write_bytes(png)
        names.append(name)
    payload = {
        "metrics": result.metrics,
        "reward_curve": result.reward_curve,
        "video_path": str(result.video_path) if result.video_path else None,
        "frames": names,
    }
    (out_dir / RESULT_FILE).write_text(json.dumps(payload))


SERVE_POLL_S = 0.1
SERVE_MAX_WAIT_S = 3600.0  # an unused warm process gives up after this long


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("reward", type=Path, nargs="?")
    ap.add_argument("out_dir", type=Path, nargs="?")
    ap.add_argument("--preset", default="round")
    ap.add_argument("--impl", default=None, help="jax or warp (default: warp on a GPU)")
    ap.add_argument("--scene", type=Path, default=None, help="scene spec JSON (default: plain flat floor)")
    ap.add_argument("--fake", action="store_true", help="use the no-GPU fake simulator (tests)")
    ap.add_argument("--serve", type=Path, default=None, help="warm up, then run the job written to this file")
    ap.add_argument("--parent-pid", type=int, default=None, help="with --serve: give up if this process exits")
    ap.add_argument("--behavior-metrics", action="store_true", help="report behavior at every eval (go1.Go1Env.behavior)")
    args = ap.parse_args(argv)

    if args.serve:
        _warm_up(args.fake)
        job = _wait_for_job(args.serve, args.parent_pid or os.getppid())
        if job is None:
            return 0
        args.reward, args.out_dir = Path(job["reward"]), Path(job["out_dir"])
        args.preset, args.impl = job.get("preset", args.preset), job.get("impl")
        args.behavior_metrics = bool(job.get("behavior_metrics"))
        args.scene = Path(job["scene"]) if job.get("scene") else None
        args.out_dir.mkdir(parents=True, exist_ok=True)
        _redirect_output(args.out_dir / "train.log")
    elif args.reward is None or args.out_dir is None:
        ap.error("reward and out_dir are required unless --serve is given")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    try:
        from .scene import parse

        scene = parse(args.scene.read_text()) if args.scene else None
        sim = _simulator(args.fake, args.preset, args.impl, args.behavior_metrics)
        result = sim.run(args.reward.read_text(), args.out_dir, scene)
    except Exception as exc:
        (args.out_dir / RESULT_FILE).write_text(
            json.dumps({"error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()})
        )
        traceback.print_exc()
        return 1
    write_result(args.out_dir, result)
    return 0


def _simulator(fake: bool, preset: str, impl: str | None, behavior_metrics: bool = False):
    if fake:
        from .fake import FakeSimulator

        return FakeSimulator()
    from .playground import PlaygroundSimulator

    return PlaygroundSimulator(preset=preset, impl=impl, behavior_metrics=behavior_metrics)


def _warm_up(fake: bool) -> None:
    """Pay for the imports and the GPU client before the job arrives."""
    if fake:
        from . import fake  # noqa: F401
        return
    import jax

    from . import playground  # noqa: F401

    jax.devices()


def _wait_for_job(path: Path, parent: int) -> dict | None:
    """The job, once parallel.py writes it; None if the parent went away or never sent one."""
    start = time.monotonic()
    while not path.exists():
        if not _alive(parent) or time.monotonic() - start > SERVE_MAX_WAIT_S:
            return None
        time.sleep(SERVE_POLL_S)
    return json.loads(path.read_text())


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        pass
    return True


def _redirect_output(log: Path) -> None:
    """Send this process's stdout and stderr (including C libraries') to the job's train.log."""
    sys.stdout.flush()
    sys.stderr.flush()
    fd = os.open(log, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    os.dup2(fd, 1)
    os.dup2(fd, 2)
    os.close(fd)


if __name__ == "__main__":
    sys.exit(main())
