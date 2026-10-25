"""MuJoCo Playground backend: the real simulator behind the worker.

Settings (env vars):
  REWARDSMITH_PRESET  smoke | round | final   (default round; use smoke on a laptop)
  REWARDSMITH_IMPL    jax | warp              (default: warp on a CUDA GPU, jax elsewhere)
  REWARDSMITH_PARALLEL  candidates trained at once, one process each (default 4)
  REWARDSMITH_RANDOMIZE 1 = train on domain-randomized robots (default 1: about 5% slower,
                        same learning within a round, and spin in place falls less)
  REWARDSMITH_PUSH      1 = train with random pushes (default 0)
  REWARDSMITH_ROBUSTNESS  1 = score the policy on unseen robots, seeds and pushes (default 1)
  REWARDSMITH_XLA_AUTOTUNE  1 to keep XLA's GPU autotuner on (off by default, see below)
"""

import logging
import os
import threading
from pathlib import Path

# The worker gates rewards on the GPU itself, so it must not grab 75% of GPU memory
# before the training processes start. Has to be set before jax initializes.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
# XLA's GPU autotuner times candidate kernels for every matmul while compiling.
# With four training processes compiling on one GPU that cost about a minute a
# round, and switching it off left steps/s and rewards unchanged on the 4090.
# The training processes inherit this. Set REWARDSMITH_XLA_AUTOTUNE=1 to keep it.
if os.environ.get("REWARDSMITH_XLA_AUTOTUNE") != "1" and "xla_gpu_autotune_level" not in os.environ.get("XLA_FLAGS", ""):
    os.environ["XLA_FLAGS"] = f"{os.environ.get('XLA_FLAGS', '')} --xla_gpu_autotune_level=0".strip()

from . import evaluation, gate, go1, go1_spec, parallel, train  # noqa: E402
from .base import Cull, RunResult, Stopped  # noqa: E402
from .progress import ProgressWriter  # noqa: E402
from .scene import EMPTY, SceneSpec  # noqa: E402

log = logging.getLogger("rewardsmith.sim")


class PlaygroundSimulator:
    def __init__(
        self,
        env_name: str = go1.BASE_ENV,
        preset: str | None = None,
        impl: str | None = None,
        randomize: bool | None = None,
        push: bool | None = None,
        robustness: bool | None = None,
        behavior_metrics: bool = False,
        evaluation_spec: list[dict] | None = None,
    ):
        if env_name != go1.BASE_ENV:
            raise ValueError(f"only {go1.BASE_ENV} is supported so far, got {env_name}")
        self.preset = preset or os.environ.get("REWARDSMITH_PRESET", "round")
        self.impl = impl or go1.default_impl()
        self.max_parallel = int(os.environ.get("REWARDSMITH_PARALLEL", "4"))
        self.randomize = _flag("REWARDSMITH_RANDOMIZE", "1") if randomize is None else randomize
        self.push = _flag("REWARDSMITH_PUSH") if push is None else push
        self.robustness = _flag("REWARDSMITH_ROBUSTNESS", "1") if robustness is None else robustness
        self.evaluation_spec = evaluation_spec
        self.behavior_metrics = behavior_metrics  # see go1.Go1Env.behavior
        self._pool: parallel.WarmPool | None = None
        # The worker gates drafts in threads, but two warp rollouts at once in one
        # process crash: warp captures a CUDA graph on its first call, and any other
        # GPU work during the capture fails it (CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED).
        self._gate_lock = threading.Lock()

    def describe(self, scene: SceneSpec | None = None) -> str:
        return go1_spec.describe(scene or EMPTY)

    def gate(self, source: str, scene: SceneSpec | None = None) -> str | None:
        """Gate in a warm training process when one is free, else here.

        In a warm process the gates of a round run side by side instead of one at a
        time, on a process whose imports and first compile are already paid, and the
        process that passed a source then trains it (see parallel.WarmPool).
        """
        pool = self._pool
        warm = pool.reserve() if pool else None
        if warm is not None:
            try:
                error = warm.gate(source, scene, self.impl)
            except parallel.WarmProcessError:
                log.exception("warm gate failed; gating in the worker instead")
                pool.discard(warm)
            else:
                pool.release(warm)
                return error
        with self._gate_lock:
            return gate.check_reward(source, impl=self.impl, scene=scene)

    def run(self, source: str, out_dir: Path, scene: SceneSpec | None = None) -> RunResult:
        progress = ProgressWriter(out_dir, train.ppo_kwargs(self.preset)["num_timesteps"])
        progress.stage("compiling")
        progress.start_heartbeat()  # nothing else writes during the minute of JIT compile
        try:
            return self._run(source, out_dir, scene, progress)
        finally:
            progress.stop_heartbeat()

    def _run(self, source: str, out_dir: Path, scene: SceneSpec | None, progress: ProgressWriter) -> RunResult:
        env = go1.make_env(source, impl=self.impl, scene=scene, behavior_metrics=self.behavior_metrics)
        # Pushes are a training condition only: the video and metrics come from a push-free env.
        # The training env keeps the behavior metrics too: halving ranks on the training curve.
        train_env = (
            go1.make_env(
                source, impl=self.impl, overrides=go1.PUSH_OVERRIDES, scene=scene, behavior_metrics=self.behavior_metrics
            )
            if self.push
            else env
        )
        result = train.train(train_env, self.preset, on_progress=progress.point, randomize=self.randomize)
        progress.stage("rendering")
        # The video, target metrics and robustness rollouts, all in one batch (see evaluation.py).
        video, frames, rollout, robust = evaluation.evaluate(env, result, out_dir, robust=self.robustness, success_spec=self.evaluation_spec)
        (out_dir / "reward.py").write_text(source)

        last = result.curve[-1]
        metrics = {
            "episode_reward": round(last.episode_reward, 3),
            **{f"term_{k}": round(v, 3) for k, v in last.terms.items()},
            **{k: round(v, 3) for k, v in rollout.items()},
            **robust,
            "train_s": round(result.train_s, 1),
            "compile_s": round(result.compile_s, 1),
            "train_randomized": int(self.randomize),
            "train_pushes": int(self.push),
        }
        progress.stage("done")
        return RunResult(metrics=metrics, frames=frames, video_path=video, reward_curve=result.reward_curve)

    def set_evaluation_spec(self, spec: list[dict] | None) -> None:
        self.evaluation_spec = spec

    def warm_up(self, n: int) -> None:
        """Start n training processes now, so their imports overlap the writer's Claude call."""
        if self._pool is None:
            self._pool = parallel.WarmPool(n)

    def close_warm_pool(self) -> None:
        """Release unused trainers, including rounds that never reach run_many."""
        pool, self._pool = self._pool, None
        if pool:
            pool.close()

    def run_many(
        self, jobs: list[tuple[str, Path]], scene: SceneSpec | None = None, cull: Cull | None = None
    ) -> list[RunResult | Stopped | str]:
        pool, self._pool = self._pool, None
        # Halving compares candidates at the same eval, so they all train at once.
        max_parallel = max(self.max_parallel, len(jobs)) if cull else self.max_parallel
        try:
            return parallel.run_in_processes(
                jobs, preset=self.preset, impl=self.impl, max_parallel=max_parallel, scene=scene, pool=pool,
                cull=cull, behavior_metrics=cull is not None or self.behavior_metrics,
                sim_options={"randomize": self.randomize, "push": self.push, "robustness": self.robustness,
                             "evaluation_spec": self.evaluation_spec},
            )
        finally:
            if pool:
                pool.close()


def _flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip() == "1"
