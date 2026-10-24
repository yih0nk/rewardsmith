"""MuJoCo Playground backend: the real simulator behind the worker.

Settings (env vars):
  REWARDSMITH_PRESET  smoke | round | final   (default round; use smoke on a laptop)
  REWARDSMITH_IMPL    jax | warp              (default: warp on a CUDA GPU, jax elsewhere)
  REWARDSMITH_PARALLEL  candidates trained at once, one process each (default 4)
"""

import os
import threading
from pathlib import Path

# The worker gates rewards on the GPU itself, so it must not grab 75% of GPU memory
# before the training processes start. Has to be set before jax initializes.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from . import gate, go1, go1_spec, parallel, render, train  # noqa: E402
from .base import Cull, RunResult, Stopped  # noqa: E402
from .progress import ProgressWriter  # noqa: E402


class PlaygroundSimulator:
    def __init__(
        self, env_name: str = go1.BASE_ENV, preset: str | None = None, impl: str | None = None,
        behavior_metrics: bool = False,
    ):
        if env_name != go1.BASE_ENV:
            raise ValueError(f"only {go1.BASE_ENV} is supported so far, got {env_name}")
        self.preset = preset or os.environ.get("REWARDSMITH_PRESET", "round")
        self.impl = impl or go1.default_impl()
        self.max_parallel = int(os.environ.get("REWARDSMITH_PARALLEL", "4"))
        self.behavior_metrics = behavior_metrics  # see go1.Go1Env.behavior
        self._pool: parallel.WarmPool | None = None
        # The worker gates drafts in threads, but two warp rollouts at once in one
        # process crash: warp captures a CUDA graph on its first call, and any other
        # GPU work during the capture fails it (CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED).
        self._gate_lock = threading.Lock()

    def describe(self) -> str:
        return go1_spec.describe()

    def gate(self, source: str) -> str | None:
        with self._gate_lock:
            return gate.check_reward(source, impl=self.impl)

    def run(self, source: str, out_dir: Path) -> RunResult:
        progress = ProgressWriter(out_dir, train.ppo_kwargs(self.preset)["num_timesteps"])
        progress.stage("compiling")
        env = go1.make_env(source, impl=self.impl, behavior_metrics=self.behavior_metrics)
        result = train.train(env, self.preset, on_progress=progress.point)
        progress.stage("rendering")
        video, frames, rollout = render.rollout_and_render(env, result, out_dir)
        (out_dir / "reward.py").write_text(source)

        last = result.curve[-1]
        metrics = {
            "episode_reward": round(last.episode_reward, 3),
            **{f"term_{k}": round(v, 3) for k, v in last.terms.items()},
            **{k: round(v, 3) for k, v in rollout.items()},
            "train_s": round(result.train_s, 1),
            "compile_s": round(result.compile_s, 1),
        }
        progress.stage("done")
        return RunResult(metrics=metrics, frames=frames, video_path=video, reward_curve=result.reward_curve)

    def warm_up(self, n: int) -> None:
        """Start n training processes now, so their imports overlap the writer's Claude call."""
        if self._pool is None:
            self._pool = parallel.WarmPool(n)

    def run_many(self, jobs: list[tuple[str, Path]], cull: Cull | None = None) -> list[RunResult | Stopped | str]:
        pool, self._pool = self._pool, None
        # Halving compares candidates at the same eval, so they all train at once.
        max_parallel = max(self.max_parallel, len(jobs)) if cull else self.max_parallel
        try:
            return parallel.run_in_processes(
                jobs, preset=self.preset, impl=self.impl, max_parallel=max_parallel, pool=pool, cull=cull,
                behavior_metrics=cull is not None,  # the cull ranks candidates by them
            )
        finally:
            if pool:
                pool.close()
