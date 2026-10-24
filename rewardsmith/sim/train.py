"""PPO training for a Go1Env, with timing and a per-term reward curve.

Presets trade quality for time. "smoke" is for a laptop CPU, "round" for one
search round on the GPU, "final" for the winning reward. Tune "round" from the
measured run time: it sets how many candidates and rounds fit in the weekend.
"""

import functools
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from . import jax_compat  # noqa: F401  (must come before brax's PPO trainer)
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from mujoco_playground import wrapper
from mujoco_playground.config import locomotion_params

from .go1 import BASE_ENV, Go1Env

# Overrides on top of Playground's tuned Go1 joystick PPO config.
# brax requires batch_size * num_minibatches to be a multiple of num_envs.
PRESETS: dict[str, dict[str, Any]] = {
    "smoke": dict(
        num_timesteps=60_000, num_envs=32, batch_size=32, num_minibatches=4, unroll_length=10,
        num_updates_per_batch=2, episode_length=200, num_evals=3, num_eval_envs=16, num_resets_per_eval=0,
    ),
    # On the 4090 the walk-backwards reward was at 94% of its final score by 6.5M steps.
    # brax rounds the step count up to whole eval blocks.
    "round": dict(num_timesteps=10_000_000, num_evals=5, num_eval_envs=128),
    "final": dict(num_timesteps=100_000_000, num_evals=10),
}


@dataclass
class CurvePoint:
    steps: int
    wall_s: float
    episode_reward: float
    terms: dict[str, float]
    # Go1Env.behavior() averaged over each eval episode, plus its mean length in steps
    # (below 1000 means some robots fell). Comparable across rewards.
    behavior: dict[str, float] = field(default_factory=dict)


@dataclass
class TrainResult:
    make_inference_fn: Callable[..., Any]
    params: Any
    curve: list[CurvePoint] = field(default_factory=list)
    compile_s: float = 0.0  # start to first eval, mostly JIT
    train_s: float = 0.0  # total wall time
    num_timesteps: int = 0

    @property
    def steps_per_s(self) -> float:
        run = self.train_s - self.compile_s
        return self.num_timesteps / run if run > 0 else 0.0

    @property
    def reward_curve(self) -> list[float]:
        return [p.episode_reward for p in self.curve]


def ppo_kwargs(preset: str = "round", overrides: Mapping[str, Any] | None = None) -> dict[str, Any]:
    cfg = locomotion_params.brax_ppo_config(BASE_ENV)
    kwargs = {k: v for k, v in cfg.items() if k != "network_factory"}
    kwargs.update(PRESETS[preset])
    kwargs.update(overrides or {})
    if (kwargs["batch_size"] * kwargs["num_minibatches"]) % kwargs["num_envs"]:
        raise ValueError("batch_size * num_minibatches must be a multiple of num_envs")
    kwargs["network_factory"] = functools.partial(
        ppo_networks.make_ppo_networks, **dict(cfg.network_factory)
    )
    return kwargs


def train(
    env: Go1Env,
    preset: str = "round",
    overrides: Mapping[str, Any] | None = None,
    seed: int = 0,
    on_progress: Callable[[CurvePoint], None] | None = None,
) -> TrainResult:
    kwargs = ppo_kwargs(preset, overrides)
    start = time.perf_counter()
    curve: list[CurvePoint] = []

    def progress(steps: int, metrics: Mapping[str, Any]) -> None:
        point = CurvePoint(
            steps=int(steps),
            wall_s=time.perf_counter() - start,
            episode_reward=float(metrics["eval/episode_reward"]),
            terms=_eval_group(metrics, "eval/episode_reward/"),
            behavior=_behavior(metrics),
        )
        curve.append(point)
        if on_progress:
            on_progress(point)

    make_inference_fn, params, _ = ppo.train(
        environment=env,
        wrap_env_fn=wrapper.wrap_for_brax_training,
        progress_fn=progress,
        seed=seed,
        **kwargs,
    )
    total = time.perf_counter() - start
    return TrainResult(
        make_inference_fn=make_inference_fn,
        params=params,
        curve=curve,
        compile_s=curve[0].wall_s if curve else 0.0,
        train_s=total,
        num_timesteps=int(kwargs["num_timesteps"]),
    )


def _eval_group(metrics: Mapping[str, Any], prefix: str) -> dict[str, float]:
    return {k[len(prefix):]: float(v) for k, v in metrics.items() if k.startswith(prefix) and not k.endswith("_std")}


def _behavior(metrics: Mapping[str, Any]) -> dict[str, float]:
    """Per-step means of the behavior metrics; brax reports their per-episode sums."""
    length = float(metrics.get("eval/avg_episode_length", 0.0))
    sums = _eval_group(metrics, "eval/episode_behavior/")
    out = {k: v / length for k, v in sums.items()} if length > 0 else {}
    if sums:
        out["episode_length"] = length
    return out
