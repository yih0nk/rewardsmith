"""MuJoCo Playground backend: the real simulator behind the worker.

Settings (env vars):
  REWARDSMITH_PRESET  smoke | round | final   (default round; use smoke on a laptop)
  REWARDSMITH_IMPL    jax | warp              (default: warp on a CUDA GPU, jax elsewhere)
  REWARDSMITH_PARALLEL  candidates trained at once, one process each (default 4)
"""

import os
from pathlib import Path

# The worker gates rewards on the GPU itself, so it must not grab 75% of GPU memory
# before the training processes start. Has to be set before jax initializes.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from . import gate, go1, go1_spec, parallel, render, scene_eval, train  # noqa: E402
from .base import RunResult  # noqa: E402
from .progress import ProgressWriter  # noqa: E402
from .scene import EMPTY, SceneSpec  # noqa: E402


class PlaygroundSimulator:
    def __init__(self, env_name: str = go1.BASE_ENV, preset: str | None = None, impl: str | None = None):
        if env_name != go1.BASE_ENV:
            raise ValueError(f"only {go1.BASE_ENV} is supported so far, got {env_name}")
        self.preset = preset or os.environ.get("REWARDSMITH_PRESET", "round")
        self.impl = impl or go1.default_impl()
        self.max_parallel = int(os.environ.get("REWARDSMITH_PARALLEL", "4"))

    def describe(self, scene: SceneSpec | None = None) -> str:
        return go1_spec.describe(scene or EMPTY)

    def gate(self, source: str, scene: SceneSpec | None = None) -> str | None:
        return gate.check_reward(source, impl=self.impl, scene=scene)

    def run(self, source: str, out_dir: Path, scene: SceneSpec | None = None) -> RunResult:
        progress = ProgressWriter(out_dir, train.ppo_kwargs(self.preset)["num_timesteps"])
        progress.stage("compiling")
        env = go1.make_env(source, impl=self.impl, scene=scene)
        result = train.train(env, self.preset, on_progress=progress.point)
        progress.stage("rendering")
        video, frames, rollout = render.rollout_and_render(env, result, out_dir)
        rollout.update(scene_eval.target_metrics(env, result.make_inference_fn, result.params))
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

    def run_many(self, jobs: list[tuple[str, Path]], scene: SceneSpec | None = None) -> list[RunResult | str]:
        return parallel.run_in_processes(
            jobs, preset=self.preset, impl=self.impl, max_parallel=self.max_parallel, scene=scene
        )
