"""Unitree G1 humanoid environment whose reward is a generated module.

Built on Playground's G1 joystick env (G1JoystickFlatTerrain): same robot,
observations and physics. As in go1.py, its reward terms are replaced with the
generated ones, the joystick command is zeroed, Playground's tuned gait terms can
ride under the generated ones (g1_spec.STOCK_SCALES), and steer mode lets the
policy command a frozen pretrained walker (scripts/train_walker.py --env g1).

Differences from the Go1: the stock termination only fires once the torso is
upside down, so a "fell" rule is added (torso up axis below g1_spec.FALL_UP_Z or
pelvis below FALL_HEIGHT); the stock reward is not clipped, ours is clipped below
at 0 like the Go1's; the walker's observation carries the command at [9:12], not
at the end, and a gait clock (the phase) that the env keeps advancing. Random
pushes are off unless push_overrides turns them on. No scene objects yet: only
the flat or rough floor.
"""

import os
from collections.abc import Mapping
from typing import Any

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from ml_collections import config_dict
from mujoco import mjx
from mujoco_playground import registry
from mujoco_playground._src import mjx_env
from mujoco_playground._src.locomotion.g1 import joystick

from . import warp_quiet  # noqa: F401  (silences warp's per-step solver warning)
from .bodies.base import RewardError
from .g1_spec import (
    COMMAND_LIMITS, COMMAND_SMOOTHING, FALL_HEIGHT, FALL_UP_Z, NU, STOCK_PREFIX, STOCK_PROFILES, STOCK_SCALES,
    STOCK_WEIGHT, features_for,
)
from .go1 import RewardFn, default_impl
from .reward_slot import load_reward
from .scene import EMPTY, SceneSpec

BASE_ENV = "G1JoystickFlatTerrain"
ROBOT_NQ, ROBOT_NV = 36, 35  # free joint + 29 joints
WALKER_STATE = 103  # the joystick "state" observation the walker reads
WALKER_COMMAND = slice(9, 12)  # where the command sits in it
OFF_FLOOR = 0.06  # m; a foot site higher than this is off the floor (standing, it is at 3.3 cm)


class G1Env(joystick.Joystick):
    def __init__(
        self,
        reward_fn: RewardFn,
        term_names: tuple[str, ...],
        config: config_dict.ConfigDict,
        scene: SceneSpec = EMPTY,
        behavior_metrics: bool = False,
        stock: Mapping[str, float] | None = None,
        walker: Any = None,
    ):
        self._reward_fn = reward_fn
        self._term_names = term_names
        self._stock = tuple(stock or ())
        self._walker = walker_policy(walker) if walker is not None else None
        self._scene = scene
        self._behavior_metrics = behavior_metrics
        super().__init__(task="rough_terrain" if scene.terrain == "rough" else "flat_terrain", config=config)

    def _post_init(self) -> None:
        super()._post_init()
        self._pelvis_body_id = self.mj_model.body("pelvis").id

    @property
    def action_size(self) -> int:
        return len(COMMAND_LIMITS) if self._walker is not None else super().action_size

    @property
    def steers(self) -> bool:
        return self._walker is not None

    @property
    def term_names(self) -> tuple[str, ...]:
        return self._term_names

    @property
    def scene(self) -> SceneSpec:
        return self._scene

    # The loop reads these frame-less, like the Go1's (docs/bodies.md, the env duck type): the pelvis
    # is the base. Playground's G1 code passes the frame explicitly, so it is unaffected.
    def get_local_linvel(self, data: mjx.Data, frame: str = "pelvis") -> jax.Array:
        return super().get_local_linvel(data, frame)

    def get_gyro(self, data: mjx.Data, frame: str = "pelvis") -> jax.Array:
        return super().get_gyro(data, frame)

    def get_upvector(self, data: mjx.Data) -> jax.Array:
        return self.get_gravity(data, "torso")  # Playground's "gravity" sensor here is the torso's up axis

    def features(self, data: mjx.Data, info: dict[str, Any], contact: jax.Array) -> dict[str, jax.Array]:
        return {
            "base_lin_vel": self.get_local_linvel(data, "pelvis"),
            "base_ang_vel": self.get_gyro(data, "pelvis"),
            "gravity": data.site_xmat[self._pelvis_imu_site_id].T @ jnp.array([0.0, 0.0, -1.0]),
            "up_z": self.get_upvector(data)[2],
            "base_height": data.qpos[2],
            "torso_height": data.xpos[self._torso_body_id][2],
            "joint_pos": data.qpos[7:ROBOT_NQ] - self._default_pose,
            "joint_vel": data.qvel[6:ROBOT_NV],
            "foot_contact": contact.astype(jnp.float32),
            "feet_air_time": info["feet_air_time"],
            "feet_height": data.site_xpos[self._feet_site_id][:, 2],
            "hand_height": data.site_xpos[self._hands_site_id][:, 2],
            "torques": data.actuator_force,
            "base_pos": data.qpos[0:3],
            "base_yaw": self._yaw(data),
            "time": jnp.asarray(data.time, dtype=jnp.float32),
        }

    def _yaw(self, data: mjx.Data) -> jax.Array:
        forward = data.xmat[self._pelvis_body_id][:, 0]
        return jnp.arctan2(forward[1], forward[0])

    def _get_termination(self, data: mjx.Data) -> jax.Array:
        fell = (self.get_upvector(data)[2] < FALL_UP_Z) | (data.qpos[2] < FALL_HEIGHT)
        return super()._get_termination(data) | fell

    def reset(self, rng: jax.Array) -> mjx_env.State:
        state = super().reset(rng)
        if self._behavior_metrics:
            state.metrics.update(self.behavior(state.data))
        return state

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        if self._walker is not None:
            state, action = self._steer(state, action)
        state = super().step(state, action)
        state = state.replace(reward=jnp.maximum(state.reward, 0.0))  # like the Go1's: clipped below at 0
        if self._behavior_metrics:
            state.metrics.update(self.behavior(state.data))
        return state

    def _steer(self, state: mjx_env.State, action: jax.Array) -> tuple[mjx_env.State, jax.Array]:
        """Steer mode: action -> smoothed command -> the walker's joint action for this step."""
        target = jnp.clip(action, -1.0, 1.0) * jnp.asarray(COMMAND_LIMITS)
        command = state.info["command"] + COMMAND_SMOOTHING * (target - state.info["command"])
        state.info["command"] = command
        # Joystick resamples the command once its step counter passes 500; keep ours.
        state.info["step"] = jnp.zeros_like(state.info["step"])
        walker_obs = state.obs["state"][:WALKER_STATE].at[WALKER_COMMAND].set(command)
        joints, _ = self._walker({"state": walker_obs}, jax.random.PRNGKey(0))  # deterministic: the key is unused
        return state, joints

    def behavior(self, data: mjx.Data) -> dict[str, jax.Array]:
        """Same keys as Go1Env.behavior (see there), from the pelvis and torso."""
        lin = self.get_local_linvel(data, "pelvis")
        return {
            "behavior/forward_velocity": lin[0],
            "behavior/lateral_velocity": lin[1],
            "behavior/yaw_rate": self.get_gyro(data, "pelvis")[2],
            "behavior/base_height": data.qpos[2],
            "behavior/upright": self.get_upvector(data)[2],
        }

    def _get_reward(self, data, action, info, metrics, done, first_contact, contact):
        terms = self._reward_fn(self.features(data, info, contact), action, info["last_act"])
        out = {k: jnp.asarray(terms[k], dtype=jnp.float32).reshape(()) for k in self._term_names}
        if self._stock:
            # A nonzero command opens the "only while walking" gates; no term read here uses its value
            # except joint_deviation_hip, which relaxes hip roll only for a sideways command.
            moving = {**info, "command": jnp.array([1.0, 0.0, 0.0])}
            stock = super()._get_reward(data, action, moving, metrics, done, first_contact, contact)
            out.update({STOCK_PREFIX + k: stock[k] for k in self._stock})
        return out

    def pose_metrics(self, qpos: np.ndarray) -> dict[str, float]:
        """Gait and pose metrics for one rollout (g1_spec.POSE_METRICS), from qpos per step
        already cut to the steps that count (sim/pose.py)."""
        qpos = np.asarray(qpos)
        if len(qpos) == 0:
            return {}
        model, data = self.mj_model, mujoco.MjData(self.mj_model)
        feet = np.asarray(self._feet_site_id)
        feet_z = np.empty((len(qpos), 2))
        pitch, roll = np.empty(len(qpos)), np.empty(len(qpos))
        for i, q in enumerate(qpos):
            data.qpos[:] = q
            mujoco.mj_kinematics(model, data)
            feet_z[i] = data.site_xpos[feet, 2]
            rot = data.xmat[self._torso_body_id].reshape(3, 3)
            pitch[i] = np.arcsin(np.clip(rot[2, 0], -1.0, 1.0))
            roll[i] = np.arcsin(np.clip(rot[2, 1], -1.0, 1.0))
        rough = self._scene.terrain == "rough"
        off = feet_z > OFF_FLOOR + (0.05 if rough else 0.0)
        n_off = off.sum(axis=1)
        down = ~off
        touchdowns = int((down[1:] & ~down[:-1]).sum())
        return {
            "single_support": float((n_off == 1).mean()),
            "double_support": float((n_off == 0).mean()),
            "flight": float((n_off == 2).mean()),
            "flight_held_s": _longest(n_off == 2) * self.dt,
            "step_count": float(touchdowns),
            "pitch_deg": float(np.degrees(pitch).mean()),
            "abs_roll_deg": float(np.degrees(np.abs(roll)).mean()),
        }


def _longest(flags: np.ndarray) -> int:
    best = run = 0
    for f in flags:
        run = run + 1 if f else 0
        best = max(best, run)
    return best


def walker_policy(params: Any):
    """The G1 walker's deterministic policy from its checkpoint params (scripts/train_walker.py --env g1)."""
    from brax.training.acme import running_statistics
    from brax.training.agents.ppo import networks as ppo_networks
    from mujoco_playground.config import locomotion_params

    from . import checkpoints

    if isinstance(params, (str, os.PathLike)):
        params = checkpoints.load(params)
    sizes = {k: (n,) for k, n in checkpoints.obs_sizes(params).items()}
    if sizes.get("state") != (WALKER_STATE,):
        raise ValueError(f"not a G1 walker checkpoint: its state observation is {sizes.get('state')}, not ({WALKER_STATE},)")
    nets = ppo_networks.make_ppo_networks(
        sizes, NU, preprocess_observations_fn=running_statistics.normalize,
        **dict(locomotion_params.brax_ppo_config(BASE_ENV).network_factory),
    )
    params = jax.tree_util.tree_map(jnp.asarray, params)
    return ppo_networks.make_inference_fn(nets)(params, deterministic=True)


def dummy_state(scene: SceneSpec = EMPTY) -> dict[str, jax.Array]:
    """A zero-filled state with the real keys and shapes, standing upright, for checking a reward eagerly."""
    state = {f.name: jnp.zeros(f.shape, dtype=jnp.float32) for f in features_for(scene)}
    state["gravity"] = jnp.array([0.0, 0.0, -1.0])
    state["up_z"] = jnp.float32(1.0)
    state["base_height"] = jnp.float32(0.755)
    state["torso_height"] = jnp.float32(0.80)
    state["foot_contact"] = jnp.ones(2)
    return state


def discover_terms(reward_fn: RewardFn, scene: SceneSpec = EMPTY) -> tuple[str, ...]:
    """Call the reward once on a dummy state and return its term names, sorted."""
    zeros = jnp.zeros(NU, dtype=jnp.float32)
    try:
        terms = reward_fn(dummy_state(scene), zeros, zeros)
    except Exception as exc:
        raise RewardError(f"reward() raised on a standing robot: {type(exc).__name__}: {exc}") from exc
    if not isinstance(terms, Mapping) or not terms:
        raise RewardError("reward() must return a non-empty dict of named terms")
    for name, value in terms.items():
        if not isinstance(name, str) or not name.isidentifier():
            raise RewardError(f"term name {name!r} must be a plain identifier like 'forward_velocity'")
        if jnp.size(jnp.asarray(value)) != 1:
            raise RewardError(f"term {name!r} must be a scalar, got {jnp.size(jnp.asarray(value))} values")
    return tuple(sorted(terms))


def stock_scales(reward_fn: RewardFn, profile: str | None = "gait", weight: float = STOCK_WEIGHT) -> dict[str, float]:
    """The stock terms a reward trains with, and their scales (see go1.stock_scales)."""
    own = getattr(reward_fn, "__globals__", {}).get("STOCK_TERMS", {})
    if not isinstance(own, Mapping) or not all(isinstance(k, str) and isinstance(v, (int, float)) for k, v in own.items()):
        raise RewardError("STOCK_TERMS must be a dict of stock term name -> multiplier, e.g. {\"orientation\": 0.0}")
    unknown = sorted(set(own) - set(STOCK_SCALES))
    if unknown:
        raise RewardError(f"STOCK_TERMS has unknown terms {unknown}; the G1's stock terms are {sorted(STOCK_SCALES)}")
    if profile is None:
        return {}
    defaults = STOCK_PROFILES[profile]
    mult = {k: float(own.get(k, defaults.get(k, 1.0))) for k in STOCK_SCALES}
    return {k: weight * STOCK_SCALES[k] * m for k, m in mult.items() if m != 0.0}


def make_config(
    term_names: tuple[str, ...],
    impl: str = "jax",
    overrides: Mapping[str, Any] | None = None,
    stock: Mapping[str, float] | None = None,
):
    config = registry.get_default_config(BASE_ENV)
    scales = {k: 1.0 for k in term_names} | {STOCK_PREFIX + k: v for k, v in (stock or {}).items()}
    config.reward_config.scales = config_dict.ConfigDict(scales)
    # No joystick command: steer mode sets its own each step.
    config.lin_vel_x, config.lin_vel_y, config.ang_vel_yaw = [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]
    config.push_config.enable = False  # the body's push_overrides turns pushes on
    config.impl = impl
    for key, value in (overrides or {}).items():
        target = config
        *path, leaf = key.split(".")
        for part in path:
            target = target[part]
        target[leaf] = value
    return config


def make_env(
    source: str,
    impl: str | None = None,
    overrides: Mapping[str, Any] | None = None,
    scene: SceneSpec | None = None,
    behavior_metrics: bool = False,
    stock_profile: str | None = None,
    stock_weight: float = STOCK_WEIGHT,
    walker: Any = None,
) -> G1Env:
    """Build a G1 env from reward source code. Raises RewardError for bad rewards (see go1.make_env)."""
    impl = impl or default_impl()
    scene = scene or EMPTY
    if scene.objects:
        raise RewardError("the G1 humanoid has no scene objects yet: only a flat or rough floor")
    try:
        reward_fn = load_reward(source)
    except Exception as exc:
        raise RewardError(f"{type(exc).__name__}: {exc}") from exc
    term_names = discover_terms(reward_fn, scene)
    stock = stock_scales(reward_fn, stock_profile, stock_weight)
    clash = sorted(k for k in term_names if k.startswith(STOCK_PREFIX))
    if stock and clash:
        raise RewardError(f"term names starting with {STOCK_PREFIX!r} are taken by the stock terms: rename {clash}")
    mjx_env.ensure_menagerie_exists()
    config = make_config(term_names, impl, overrides, stock)
    return G1Env(reward_fn, term_names, config, scene, behavior_metrics, stock, walker)
