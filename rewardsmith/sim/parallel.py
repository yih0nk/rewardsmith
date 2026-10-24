"""Train several candidates at once, one process each, sharing one GPU.

Why processes: every new reward pays several minutes of JIT compile, which is
mostly CPU work. Separate processes compile in parallel, and on the GPU their
training interleaves. Each process gets a slice of GPU memory so they fit
together (JAX otherwise grabs 75% in the first process).
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

from .base import RunResult
from .run_one import FRAMES_DIR, RESULT_FILE

GPU_MEMORY_SHARE = 0.9  # split across processes; the rest stays free for the driver


def run_in_processes(
    jobs: list[tuple[str, Path]],
    *,
    preset: str = "round",
    impl: str | None = None,
    max_parallel: int = 4,
    fake: bool = False,
    timeout_s: float = 3600,
) -> list[RunResult | str]:
    """Run (source, out_dir) jobs, at most max_parallel at a time. Same order as jobs."""
    if not jobs:
        return []
    slots = max(1, min(max_parallel, len(jobs)))
    env = {
        **os.environ,
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        "XLA_PYTHON_CLIENT_MEM_FRACTION": f"{GPU_MEMORY_SHARE / slots:.2f}",
    }

    results: list[RunResult | str | None] = [None] * len(jobs)
    pending = list(enumerate(jobs))
    running: dict[int, tuple[subprocess.Popen, Path, float]] = {}

    while pending or running:
        while pending and len(running) < slots:
            i, (source, out_dir) = pending.pop(0)
            running[i] = (_start(source, out_dir, preset, impl, fake, env), out_dir, time.monotonic())
        time.sleep(0.5)
        for i, (proc, out_dir, started) in list(running.items()):
            if proc.poll() is None:
                if time.monotonic() - started > timeout_s:
                    proc.kill()
                    proc.wait()
                    results[i] = f"Timeout: training took over {timeout_s:.0f}s"
                    del running[i]
                continue
            results[i] = _collect(out_dir, proc.returncode)
            del running[i]

    return [r if r is not None else "Error: no result" for r in results]


def _start(source: str, out_dir: Path, preset: str, impl: str | None, fake: bool, env: dict) -> subprocess.Popen:
    out_dir.mkdir(parents=True, exist_ok=True)
    reward_file = out_dir / "reward.py"
    reward_file.write_text(source)
    (out_dir / RESULT_FILE).unlink(missing_ok=True)
    cmd = [sys.executable, "-m", "rewardsmith.sim.run_one", str(reward_file), str(out_dir), "--preset", preset]
    if impl:
        cmd += ["--impl", impl]
    if fake:
        cmd.append("--fake")
    log = open(out_dir / "train.log", "w")  # noqa: SIM115 (closed when the process exits)
    return subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)


def _collect(out_dir: Path, returncode: int) -> RunResult | str:
    result_file = out_dir / RESULT_FILE
    if not result_file.exists():
        tail = _log_tail(out_dir)
        return f"Process exited with code {returncode} and no result. Log tail:\n{tail}"
    data = json.loads(result_file.read_text())
    if "error" in data:
        return data["error"]
    frames = [(out_dir / FRAMES_DIR / name).read_bytes() for name in data.get("frames", [])]
    video = Path(data["video_path"]) if data.get("video_path") else None
    return RunResult(metrics=data["metrics"], frames=frames, video_path=video, reward_curve=data["reward_curve"])


def _log_tail(out_dir: Path, lines: int = 15) -> str:
    log = out_dir / "train.log"
    if not log.exists():
        return "(no log)"
    return "\n".join(log.read_text(errors="replace").splitlines()[-lines:])
