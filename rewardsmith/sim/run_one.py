"""Train one reward in its own process. parallel.py starts one of these per candidate.

    python -m rewardsmith.sim.run_one <reward.py> <out_dir> [--preset round] [--impl jax] [--scene scene.json] [--fake]
        [--init-params params.pkl] [--steps N]
    python -m rewardsmith.sim.run_one --serve <job.json> [--fake]

Writes <out_dir>/result.json, plus frames/NN.png for the critic, and the
policy's params at every eval (params_<steps>.pkl, see checkpoints.py). On
failure, result.json holds {"error": "..."} and the exit code is 1.

--init-params starts training from a checkpoint (any params_<steps>.pkl, or the
pretrained walker) instead of the default warm start; --steps trains that many
steps instead of the preset's. Together they continue any job's policy under a
new reward (docs/warm-start.md).

--serve starts a warm process: it does the slow imports and a throwaway gate
first (~30 s on the pod, overlapping the writer's Claude call), then answers
parallel.py's messages in job.json:
  {"kind": "gate", "seq", "source", "scene", "impl", "stock"}  gate the source and write
      {"seq", "error"} (or {"seq", "crash"} for a non-reward failure) to job.gate.json;
      then wait for the next message. The orchestrator's fix-retry loop reuses this.
  {"kind": "train", "reward", "out_dir", "preset", "impl", "init_params", "steps", ...}  run that job and exit.
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
    ap.add_argument("--sim-options", type=json.loads, default={}, help="resolved training and robustness flags")
    ap.add_argument("--init-params", type=Path, default=None, help="checkpoint to start training from")
    ap.add_argument("--steps", type=int, default=None, help="training steps (default: the preset's)")
    args = ap.parse_args(argv)

    if args.serve:
        _warm_up(args.fake)
        job = _serve_gates(args.serve, args.parent_pid or os.getppid(), args.fake)
        if job is None:
            return 0
        args.reward, args.out_dir = Path(job["reward"]), Path(job["out_dir"])
        args.preset, args.impl = job.get("preset", args.preset), job.get("impl")
        args.behavior_metrics = bool(job.get("behavior_metrics"))
        args.sim_options = job.get("sim_options") or {}
        args.scene = Path(job["scene"]) if job.get("scene") else None
        args.init_params = Path(job["init_params"]) if job.get("init_params") else None
        args.steps = job.get("steps")
        args.out_dir.mkdir(parents=True, exist_ok=True)
        _redirect_output(args.out_dir / "train.log")
    elif args.reward is None or args.out_dir is None:
        ap.error("reward and out_dir are required unless --serve is given")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    try:
        from .scene import parse

        scene = parse(args.scene.read_text()) if args.scene else None
        sim = _simulator(args.fake, args.preset, args.impl, args.behavior_metrics, args.sim_options)
        start = {"init_params": args.init_params, "steps": args.steps}
        result = sim.run(args.reward.read_text(), args.out_dir, scene, **{k: v for k, v in start.items() if v})
    except Exception as exc:
        (args.out_dir / RESULT_FILE).write_text(
            json.dumps({"error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()})
        )
        traceback.print_exc()
        return 1
    write_result(args.out_dir, result)
    return 0


def _simulator(
    fake: bool, preset: str, impl: str | None, behavior_metrics: bool = False,
    sim_options: dict[str, object] | None = None,
):
    if fake:
        from .fake import FakeSimulator

        sim = FakeSimulator()
        sim.set_evaluation_spec((sim_options or {}).get("evaluation_spec"))
        return sim
    from .playground import PlaygroundSimulator

    return PlaygroundSimulator(preset=preset, impl=impl, behavior_metrics=behavior_metrics, **(sim_options or {}))


# A reward every scene can run: the warm-up gates it to pay for first-use costs early.
WARM_UP_REWARD = """import jax.numpy as jnp


def reward(state, action, prev_action):
    return {"still": -jnp.sum(jnp.square(action))}
"""


def _warm_up(fake: bool) -> None:
    """Pay for the imports, the GPU client and first-use costs before the job arrives.

    Gating a throwaway reward loads the Go1 model and warp's kernels and runs XLA's
    first compile, about 5 s that the first real gate would otherwise pay.
    """
    if fake:
        from . import fake  # noqa: F401
        return
    import jax

    from . import gate, playground  # noqa: F401

    jax.devices()
    try:
        error = gate.check_reward(WARM_UP_REWARD)
    except Exception:  # a warm-up is an optimisation; the real gate reports real problems
        traceback.print_exc()
    else:
        if error:
            print(f"warm-up gate failed: {error}", flush=True)


def _serve_gates(path: Path, parent: int, fake: bool) -> dict | None:
    """Answer gate messages until the train job arrives; return it (None: none came)."""
    while True:
        job = _wait_for_job(path, parent)
        if job is None or job.get("kind", "train") == "train":
            return job
        answer: dict = {"seq": job["seq"]}
        try:
            answer["error"] = _gate(job, fake)
        except Exception:
            answer["crash"] = traceback.format_exc()[-3000:]
        tmp = path.with_name(path.stem + ".gate.tmp")
        tmp.write_text(json.dumps(answer))
        os.replace(tmp, path.with_name(path.stem + ".gate.json"))


def _gate(job: dict, fake: bool) -> str | None:
    from .scene import parse

    scene = parse(job["scene"]) if job.get("scene") else None
    if fake:
        from .fake import FakeSimulator

        return FakeSimulator().gate(job["source"], scene)
    from . import gate

    return gate.check_reward(job["source"], impl=job.get("impl"), scene=scene, stock=job.get("stock"))


def _wait_for_job(path: Path, parent: int) -> dict | None:
    """The next message, once parallel.py writes it (taken off the file, so the next one
    can follow); None if the parent went away or never sent one."""
    start = time.monotonic()
    while not path.exists():
        if not _alive(parent) or time.monotonic() - start > SERVE_MAX_WAIT_S:
            return None
        time.sleep(SERVE_POLL_S)
    job = json.loads(path.read_text())
    path.unlink()
    return job


def _alive(pid: int) -> bool:
    if os.name == "nt":
        # os.kill(pid, 0) terminates processes on Windows; wait on a handle instead.
        import ctypes
        from ctypes import wintypes

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        kernel.WaitForSingleObject.restype = wintypes.DWORD
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel.CloseHandle.restype = wintypes.BOOL
        handle = kernel.OpenProcess(0x00100000, False, pid)  # SYNCHRONIZE
        if not handle:
            return ctypes.get_last_error() != 87  # invalid PID; access denied may still be alive
        try:
            return kernel.WaitForSingleObject(handle, 0) != 0
        finally:
            kernel.CloseHandle(handle)
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
