"""MuJoCo Playground backend: the real simulator behind the worker.

Settings (env vars):
  REWARDSMITH_PRESET  smoke | round | final   (default round; use smoke on a laptop)
  REWARDSMITH_IMPL    jax | warp              (default: warp on a CUDA GPU, jax elsewhere)
  REWARDSMITH_PARALLEL  candidates trained at once, one process each (default 4)
  REWARDSMITH_XLA_AUTOTUNE  1 to keep XLA's GPU autotuner on (off by default, see below)
"""

import os
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

from . import gate, go1, go1_spec, parallel, render, train  # noqa: E402
from .base import RunResult  # noqa: E402
from .progress import ProgressWriter  # noqa: E402


class PlaygroundSimulator:
    def __init__(self, env_name: str = go1.BASE_ENV, preset: str | None = None, impl: str | None = None):
        if env_name != go1.BASE_ENV:
            raise ValueError(f"only {go1.BASE_ENV} is supported so far, got {env_name}")
        self.preset = preset or os.environ.get("REWARDSMITH_PRESET", "round")
        self.impl = impl or go1.default_impl()
        self.max_parallel = int(os.environ.get("REWARDSMITH_PARALLEL", "4"))

    def describe(self) -> str:
        return go1_spec.describe()

    def gate(self, source: str) -> str | None:
        return gate.check_reward(source, impl=self.impl)

    def run(self, source: str, out_dir: Path) -> RunResult:
        progress = ProgressWriter(out_dir, train.ppo_kwargs(self.preset)["num_timesteps"])
        progress.stage("compiling")
        env = go1.make_env(source, impl=self.impl)
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

    def run_many(self, jobs: list[tuple[str, Path]]) -> list[RunResult | str]:
        return parallel.run_in_processes(jobs, preset=self.preset, impl=self.impl, max_parallel=self.max_parallel)
