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
import tempfile
import time
from pathlib import Path

from .base import Cull, RunResult, Stopped
from .progress import point_metrics, read_progress
from .run_one import FRAMES_DIR, RESULT_FILE
from .scene import SceneSpec

GPU_MEMORY_SHARE = 0.9  # split across processes; the rest stays free for the driver


class WarmPool:
    """Training processes started before their rewards exist.

    Each one runs the ~15 s of imports while the writer (or editor) is still
    waiting on Claude, then picks up a job from run_in_processes.
    """

    def __init__(self, n: int, *, fake: bool = False):
        self.dir = Path(tempfile.mkdtemp(prefix="rewardsmith_warm_"))
        self.env = _child_env(n)
        self.procs: list[tuple[subprocess.Popen, Path]] = []
        try:
            for i in range(n):
                job = self.dir / f"job{i}.json"
                cmd = [sys.executable, "-m", "rewardsmith.sim.run_one", "--serve", str(job), "--parent-pid", str(os.getpid())]
                if fake:
                    cmd.append("--fake")
                with open(self.dir / f"warm{i}.log", "w") as log:
                    self.procs.append((subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=self.env), job))
        except BaseException:
            self.close()
            raise

    def take(self) -> tuple[subprocess.Popen, Path] | None:
        """A still-running warm process and its job file, or None when they are used up."""
        while self.procs:
            proc, job = self.procs.pop(0)
            if proc.poll() is None:
                return proc, job
        return None

    def close(self) -> None:
        """Stop the warm processes no job was given to."""
        for proc, _ in self.procs:
            if proc.poll() is None:
                proc.kill()
            proc.wait()
        self.procs = []


def run_in_processes(
    jobs: list[tuple[str, Path]],
    *,
    preset: str = "round",
    impl: str | None = None,
    max_parallel: int = 4,
    fake: bool = False,
    timeout_s: float = 3600,
    scene: SceneSpec | None = None,
    pool: WarmPool | None = None,
    cull: Cull | None = None,
    behavior_metrics: bool = False,
    sim_options: dict[str, object] | None = None,
) -> list[RunResult | Stopped | str]:
    """Run (source, out_dir) jobs, at most max_parallel at a time. Same order as jobs.

    Every job trains in scene. Jobs go to pool's warm processes first, then to newly
    started ones. cull, if given, is asked every poll which running jobs to stop early.
    """
    if not jobs:
        return []
    slots = max(1, min(max_parallel, len(jobs)))
    env = _child_env(slots)

    results: list[RunResult | Stopped | str | None] = [None] * len(jobs)
    pending = list(enumerate(jobs))
    running: dict[int, tuple[subprocess.Popen, Path, float]] = {}

    while pending or running:
        while pending and len(running) < slots:
            i, (source, out_dir) = pending.pop(0)
            warm = pool.take() if pool else None
            if warm:
                proc = _hand_over(warm, source, out_dir, preset, impl, behavior_metrics, scene, sim_options)
            else:
                proc = _start(source, out_dir, preset, impl, fake, env, behavior_metrics, scene, sim_options)
            running[i] = (proc, out_dir, time.monotonic())
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
        if cull and running:
            for i, reason in cull(set(running)).items():
                if i in running:
                    proc, out_dir, _ = running.pop(i)
                    proc.kill()
                    proc.wait()
                    results[i] = _stopped(out_dir, reason)

    return [r if r is not None else "Error: no result" for r in results]


def _child_env(slots: int) -> dict[str, str]:
    return {
        **os.environ,
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        "XLA_PYTHON_CLIENT_MEM_FRACTION": f"{GPU_MEMORY_SHARE / slots:.2f}",
    }


def _prepare(source: str, out_dir: Path, scene: SceneSpec | None = None) -> tuple[Path, Path | None]:
    """Write the job's reward.py, and scene.json unless the scene is the plain floor."""
    out_dir.mkdir(parents=True, exist_ok=True)
    reward_file = out_dir / "reward.py"
    reward_file.write_text(source)
    (out_dir / RESULT_FILE).unlink(missing_ok=True)
    scene_file = None
    if scene is not None and not scene.is_empty:
        scene_file = out_dir / "scene.json"
        scene_file.write_text(scene.to_json())
    return reward_file, scene_file


def _hand_over(
    warm: tuple[subprocess.Popen, Path], source: str, out_dir: Path, preset: str, impl: str | None,
    behavior_metrics: bool = False, scene: SceneSpec | None = None,
    sim_options: dict[str, object] | None = None,
) -> subprocess.Popen:
    proc, job = warm
    reward_file, scene_file = _prepare(source, out_dir, scene)
    payload = {
        "reward": str(reward_file), "out_dir": str(out_dir), "preset": preset, "impl": impl,
        "behavior_metrics": behavior_metrics, "scene": str(scene_file) if scene_file else None,
        "sim_options": sim_options or {},
    }
    tmp = job.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload))
    os.replace(tmp, job)  # the child polls for job; it must never see half a file
    return proc


def _start(
    source: str, out_dir: Path, preset: str, impl: str | None, fake: bool, env: dict,
    behavior_metrics: bool = False, scene: SceneSpec | None = None,
    sim_options: dict[str, object] | None = None,
) -> subprocess.Popen:
    reward_file, scene_file = _prepare(source, out_dir, scene)
    cmd = [sys.executable, "-m", "rewardsmith.sim.run_one", str(reward_file), str(out_dir), "--preset", preset]
    if sim_options is not None:
        cmd += ["--sim-options", json.dumps(sim_options)]
    if impl:
        cmd += ["--impl", impl]
    if scene_file:
        cmd += ["--scene", str(scene_file)]
    if fake:
        cmd.append("--fake")
    if behavior_metrics:
        cmd.append("--behavior-metrics")
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


def _stopped(out_dir: Path, reason: str) -> Stopped:
    curve = (read_progress(out_dir) or {}).get("curve") or []
    metrics = point_metrics(curve[-1]) if curve else {}
    return Stopped(reason=reason, metrics=metrics, reward_curve=[p.get("episode_reward") for p in curve])


def _log_tail(out_dir: Path, lines: int = 15) -> str:
    log = out_dir / "train.log"
    if not log.exists():
        return "(no log)"
    return "\n".join(log.read_text(errors="replace").splitlines()[-lines:])
