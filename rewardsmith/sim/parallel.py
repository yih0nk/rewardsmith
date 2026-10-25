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
import threading
import time
from pathlib import Path

from .base import Cull, OnDone, RunResult, Stopped
from .progress import point_metrics, read_progress
from .run_one import FRAMES_DIR, RESULT_FILE
from .scene import SceneSpec

GPU_MEMORY_SHARE = 0.9  # split across processes; the rest stays free for the driver


class WarmProcessError(RuntimeError):
    """A warm process died or stopped answering. The caller falls back to gating in-process."""


GATE_TIMEOUT_S = 600.0
GATE_POLL_S = 0.05


class WarmProcess:
    """One warm training process: gate sources in it, then hand it its training job.

    The protocol is files in the pool's directory. The parent writes `job`; the child
    reads and deletes it. A gate job's answer comes back in `gate_result`, tagged with
    the job's seq. A train job is the last message: the child trains and exits.
    """

    def __init__(self, proc: subprocess.Popen, job: Path):
        self.proc, self.job = proc, job
        self.gate_result = job.with_name(job.stem + ".gate.json")
        self.gated: str | None = None  # the source that passed this process's last gate
        self._seq = 0

    def alive(self) -> bool:
        return self.proc.poll() is None

    def gate(self, source: str, scene: SceneSpec | None = None, impl: str | None = None) -> str | None:
        """Gate source in the child: None if it passes, else the message for the writer."""
        self._seq += 1
        self.gated = None
        send(self.job, {
            "kind": "gate", "seq": self._seq, "source": source, "impl": impl,
            "scene": scene.to_dict() if scene is not None and not scene.is_empty else None,
        })
        deadline = time.monotonic() + GATE_TIMEOUT_S
        while True:
            try:
                answer = json.loads(self.gate_result.read_text())
            except (OSError, ValueError):
                answer = None
            if answer is not None and answer.get("seq") == self._seq:
                break
            if not self.alive():
                raise WarmProcessError(f"warm process exited with code {self.proc.returncode} during a gate")
            if time.monotonic() > deadline:
                raise WarmProcessError(f"no gate answer after {GATE_TIMEOUT_S:.0f}s")
            time.sleep(GATE_POLL_S)
        if "crash" in answer:  # not the reward's fault: let the in-process gate decide, and raise
            raise WarmProcessError(f"gate crashed in the warm process: {answer['crash']}")
        if answer["error"] is None:
            self.gated = source
        return answer["error"]

    def kill(self) -> None:
        if self.alive():
            self.proc.kill()
        self.proc.wait()


class WarmPool:
    """Training processes started before their rewards exist.

    Each one does the imports and a throwaway compile (~30 s on the pod) while the
    writer (or editor) is still waiting on Claude. Then it gates a draft (reserve,
    WarmProcess.gate, release), and run_in_processes gives its training job to the
    process that gated that exact source, else to any free one.
    """

    def __init__(self, n: int, *, fake: bool = False):
        self.dir = Path(tempfile.mkdtemp(prefix="rewardsmith_warm_"))
        self.env = _child_env(n)
        self.procs: list[WarmProcess] = []
        self._busy: set[int] = set()  # ids of processes lent out for a gate
        self._lock = threading.Lock()  # gates run in the orchestrator's threads
        try:
            for i in range(n):
                job = self.dir / f"job{i}.json"
                cmd = [sys.executable, "-m", "rewardsmith.sim.run_one", "--serve", str(job), "--parent-pid", str(os.getpid())]
                if fake:
                    cmd.append("--fake")
                with open(self.dir / f"warm{i}.log", "w") as log:
                    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=self.env)
                self.procs.append(WarmProcess(proc, job))
        except BaseException:
            self.close()
            raise

    def reserve(self) -> WarmProcess | None:
        """A live process to gate in, holding no passed source, or None if there is none free."""
        with self._lock:
            for wp in self.procs:
                if id(wp) not in self._busy and wp.gated is None and wp.alive():
                    self._busy.add(id(wp))
                    return wp
        return None

    def release(self, wp: WarmProcess) -> None:
        """Give back a reserved process. A dead one is dropped."""
        with self._lock:
            self._busy.discard(id(wp))
            if not wp.alive() and wp in self.procs:
                self.procs.remove(wp)

    def discard(self, wp: WarmProcess) -> None:
        """Stop a reserved process that misbehaved."""
        with self._lock:
            self._busy.discard(id(wp))
            if wp in self.procs:
                self.procs.remove(wp)
        wp.kill()

    def take(self, source: str | None = None) -> WarmProcess | None:
        """The live process that gated source, else a live one with no passed source, else None."""
        with self._lock:
            live = [wp for wp in self.procs if id(wp) not in self._busy and wp.alive()]
            pick = next((wp for wp in live if source is not None and wp.gated == source), None)
            pick = pick or next((wp for wp in live if wp.gated is None), None)
            if pick is not None:
                self.procs.remove(pick)
            return pick

    def close(self) -> None:
        """Stop the warm processes no job was given to."""
        with self._lock:
            procs, self.procs = self.procs, []
        for wp in procs:
            wp.kill()


def send(path: Path, payload: dict) -> None:
    """Write a message for a warm process. It polls for the file, so it must never see half of one."""
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload))
    os.replace(tmp, path)


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
    on_done: OnDone | None = None,
) -> list[RunResult | Stopped | str]:
    """Run (source, out_dir) jobs, at most max_parallel at a time. Same order as jobs.

    Every job trains in scene. Jobs go to pool's warm processes first, then to newly
    started ones. cull, if given, is asked every poll which running jobs to stop early.
    on_done, if given, gets each job's result as soon as that job ends.
    """
    if not jobs:
        return []
    slots = max(1, min(max_parallel, len(jobs)))
    env = _child_env(slots)

    results: list[RunResult | Stopped | str | None] = [None] * len(jobs)
    pending = list(enumerate(jobs))
    running: dict[int, tuple[subprocess.Popen, Path, float]] = {}

    def finish(i: int, result: RunResult | Stopped | str) -> None:
        results[i] = result
        if on_done:
            on_done(i, result)

    while pending or running:
        while pending and len(running) < slots:
            i, (source, out_dir) = pending.pop(0)
            warm = pool.take(source) if pool else None
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
                    del running[i]
                    finish(i, f"Timeout: training took over {timeout_s:.0f}s")
                continue
            del running[i]
            finish(i, _collect(out_dir, proc.returncode))
        if cull and running:
            for i, reason in cull(set(running)).items():
                if i in running:
                    proc, out_dir, _ = running.pop(i)
                    proc.kill()
                    proc.wait()
                    finish(i, _stopped(out_dir, reason))

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
    warm: WarmProcess, source: str, out_dir: Path, preset: str, impl: str | None,
    behavior_metrics: bool = False, scene: SceneSpec | None = None,
    sim_options: dict[str, object] | None = None,
) -> subprocess.Popen:
    reward_file, scene_file = _prepare(source, out_dir, scene)
    send(warm.job, {
        "kind": "train", "reward": str(reward_file), "out_dir": str(out_dir), "preset": preset, "impl": impl,
        "behavior_metrics": behavior_metrics, "scene": str(scene_file) if scene_file else None,
        "sim_options": sim_options or {},
    })
    return warm.proc


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
