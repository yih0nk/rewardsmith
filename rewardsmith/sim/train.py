"""PPO training for a Go1Env, with timing and a per-term reward curve.

Presets trade quality for time. "smoke" is for a laptop CPU, "round" for one
search round on the GPU, "final" for the winning reward. Tune "round" from the
measured run time: it sets how many candidates and rounds fit in the weekend.
"""

import functools
from collections.abc import Mapping
from typing import Any

from . import jax_compat  # noqa: F401  (must come before brax's PPO trainer)
from brax.training.agents.ppo import networks as ppo_networks
from mujoco_playground.config import locomotion_params

from .go1 import BASE_ENV

# Overrides on top of Playground's tuned Go1 joystick PPO config.
# brax requires batch_size * num_minibatches to be a multiple of num_envs.
PRESETS: dict[str, dict[str, Any]] = {
    "smoke": dict(
        num_timesteps=60_000, num_envs=32, batch_size=32, num_minibatches=4, unroll_length=10,
        num_updates_per_batch=2, episode_length=200, num_evals=3, num_eval_envs=16, num_resets_per_eval=0,
    ),
    "round": dict(num_timesteps=20_000_000, num_evals=5, num_eval_envs=128),
    "final": dict(num_timesteps=100_000_000, num_evals=10),
}


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
