"""Go1 environment whose reward is a generated module.

Built on Playground's Go1 joystick env: same robot, observations, physics and
termination. We replace its reward terms with the generated ones, and zero its
velocity command (our behaviors don't track a joystick).
"""

import os
from collections.abc import Callable, Mapping
from typing import Any

import jax
import jax.numpy as jnp
from ml_collections import config_dict
from mujoco import mjx
from mujoco_playground import registry
from mujoco_playground._src import mjx_env
from mujoco_playground._src.locomotion.go1 import joystick

from . import warp_quiet  # noqa: F401  (silences warp's per-step solver warning)
from .go1_spec import FEATURES
from .reward_slot import load_reward

BASE_ENV = "Go1JoystickFlatTerrain"
# Config overrides that turn on Playground's random pushes: every 1-3 s a velocity kick
# of up to 3 m/s, in a random horizontal direction, lasting 0.05-0.2 s.
PUSH_OVERRIDES: dict[str, Any] = {"pert_config.enable": True}
RewardFn = Callable[[dict[str, jax.Array], jax.Array, jax.Array], Mapping[str, Any]]


class RewardError(ValueError):
    """The reward module is unusable. The message is meant for the reward writer."""


class Go1Env(joystick.Joystick):
    def __init__(
        self, reward_fn: RewardFn, term_names: tuple[str, ...], config: config_dict.ConfigDict,
        behavior_metrics: bool = False,
    ):
        self._reward_fn = reward_fn
        self._term_names = term_names
        self._behavior_metrics = behavior_metrics
        super().__init__(task="flat_terrain", config=config)

    @property
    def term_names(self) -> tuple[str, ...]:
        return self._term_names

    def features(self, data: mjx.Data, info: dict[str, Any], contact: jax.Array) -> dict[str, jax.Array]:
        feet_z = data.site_xpos[self._feet_site_id][:, 2]
        return {
            "base_lin_vel": self.get_local_linvel(data),
            "base_ang_vel": self.get_gyro(data),
            "gravity": self.get_gravity(data),
            "up_z": self.get_upvector(data)[2],
            "base_height": data.qpos[2],
            "joint_pos": data.qpos[7:] - self._default_pose,
            "joint_vel": data.qvel[6:],
            "foot_contact": contact.astype(jnp.float32),
            "feet_air_time": info["feet_air_time"],
            "feet_height": feet_z,
            "torques": data.actuator_force,
        }

    def reset(self, rng: jax.Array) -> mjx_env.State:
        state = super().reset(rng)
        if self._behavior_metrics:
            state.metrics.update(self.behavior(state.data))
        return state

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        state = super().step(state, action)
        if self._behavior_metrics:
            state.metrics.update(self.behavior(state.data))
        return state

    def behavior(self, data: mjx.Data) -> dict[str, jax.Array]:
        """What the robot is doing, reward aside. With behavior_metrics on, every eval
        reports these, summed over the episode, so candidates with different rewards
        can be compared mid-training (successive halving). Off by default: they change
        the compiled program, so default training stays exactly as it was."""
        lin = self.get_local_linvel(data)
        return {
            "behavior/forward_velocity": lin[0],
            "behavior/lateral_velocity": lin[1],
            "behavior/yaw_rate": self.get_gyro(data)[2],
            "behavior/base_height": data.qpos[2],
            "behavior/upright": self.get_upvector(data)[2],
        }

    def _get_reward(self, data, action, info, metrics, done, first_contact, contact):
        del metrics, done, first_contact
        terms = self._reward_fn(self.features(data, info, contact), action, info["last_act"])
        return {k: jnp.asarray(terms[k], dtype=jnp.float32).reshape(()) for k in self._term_names}


def dummy_state() -> dict[str, jax.Array]:
    """A zero-filled state with the real keys and shapes, for checking a reward eagerly."""
    state = {f.name: jnp.zeros(f.shape, dtype=jnp.float32) for f in FEATURES}
    state["gravity"] = jnp.array([0.0, 0.0, -1.0])
    state["up_z"] = jnp.float32(1.0)
    return state


def discover_terms(reward_fn: RewardFn) -> tuple[str, ...]:
    """Call the reward once on a dummy state and return its term names, sorted."""
    zeros = jnp.zeros(12, dtype=jnp.float32)
    try:
        terms = reward_fn(dummy_state(), zeros, zeros)
    except Exception as exc:
        raise RewardError(f"reward() raised on a standing robot: {type(exc).__name__}: {exc}") from exc
    if not isinstance(terms, Mapping) or not terms:
        raise RewardError("reward() must return a non-empty dict of named terms")
    for name, value in terms.items():
        if not isinstance(name, str) or not name.isidentifier():
            raise RewardError(f"term name {name!r} must be a plain identifier like 'backward_velocity'")
        size = jnp.size(jnp.asarray(value))
        if size != 1:
            raise RewardError(f"term {name!r} must be a scalar, got {size} values")
    return tuple(sorted(terms))


def make_config(term_names: tuple[str, ...], impl: str = "jax", overrides: Mapping[str, Any] | None = None):
    config = registry.get_default_config(BASE_ENV)
    # Must happen before the env locks its config: scales are looked up by term name.
    config.reward_config.scales = config_dict.ConfigDict({k: 1.0 for k in term_names})
    config.command_config.a = [0.0, 0.0, 0.0]  # no joystick command
    config.impl = impl
    for key, value in (overrides or {}).items():
        target = config
        *path, leaf = key.split(".")
        for part in path:
            target = target[part]
        target[leaf] = value
    return config


def default_impl() -> str:
    """REWARDSMITH_IMPL if set; else warp on a CUDA GPU (2.4x faster on the 4090), jax elsewhere."""
    return os.environ.get("REWARDSMITH_IMPL") or ("warp" if jax.default_backend() == "gpu" else "jax")


def make_env(
    source: str, impl: str | None = None, overrides: Mapping[str, Any] | None = None, behavior_metrics: bool = False
) -> Go1Env:
    """Build a Go1 env from reward source code. Raises RewardError for bad rewards."""
    impl = impl or default_impl()
    try:
        reward_fn = load_reward(source)
    except Exception as exc:
        raise RewardError(f"{type(exc).__name__}: {exc}") from exc
    term_names = discover_terms(reward_fn)
    mjx_env.ensure_menagerie_exists()
    return Go1Env(reward_fn, term_names, make_config(term_names, impl, overrides), behavior_metrics)
