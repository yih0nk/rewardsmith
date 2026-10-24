"""Go1 environment whose reward is a generated module.

Built on Playground's Go1 joystick env: same robot, observations, physics and
termination. We replace its reward terms with the generated ones, and zero its
velocity command (our behaviors don't track a joystick).

With a scene (see scene.py), the env loads the compiled scene instead of the
stock flat floor, places every object afresh at each reset, adds per-object
features for the reward, and appends each object's relative position (and the
episode time) to the policy's observation so it can learn the skill.
"""

import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from ml_collections import config_dict
from mujoco import mjx
from mujoco_playground import registry
from mujoco_playground._src import mjx_env
from mujoco.mjx._src import math as mjx_math
from mujoco_playground._src.locomotion.go1 import base as go1_base
from mujoco_playground._src.locomotion.go1 import joystick

from . import scene_xml
from . import warp_quiet  # noqa: F401  (silences warp's per-step solver warning)
from .go1_spec import features_for
from .reward_slot import load_reward
from .scene import EMPTY, SceneObject, SceneSpec

BASE_ENV = "Go1JoystickFlatTerrain"
# Config overrides that turn on Playground's random pushes: every 1-3 s a velocity kick
# of up to 3 m/s, in a random horizontal direction, lasting 0.05-0.2 s.
PUSH_OVERRIDES: dict[str, Any] = {"pert_config.enable": True}
RewardFn = Callable[[dict[str, jax.Array], jax.Array, jax.Array], Mapping[str, Any]]

# The robot's slice of qpos/qvel: free joint + 12 joints. Movable scene objects come after.
ROBOT_NQ, ROBOT_NV = 19, 18
MAX_OBS_DIST = 10.0  # m; clip object distances in the observation


class RewardError(ValueError):
    """The reward module is unusable. The message is meant for the reward writer."""


@dataclass(frozen=True)
class _Placed:
    """A scene object and where the model keeps it."""

    obj: SceneObject
    body_id: int
    mocap_id: int  # -1 for movable objects
    qpos_adr: int  # free joint address, movable objects only
    dof_adr: int


class Go1Env(joystick.Joystick):
    def __init__(
        self,
        reward_fn: RewardFn,
        term_names: tuple[str, ...],
        config: config_dict.ConfigDict,
        scene: SceneSpec = EMPTY,
        behavior_metrics: bool = False,
    ):
        self._reward_fn = reward_fn
        self._term_names = term_names
        self._scene = scene
        self._behavior_metrics = behavior_metrics
        if scene.is_empty:
            super().__init__(task="flat_terrain", config=config)
        else:
            # Joystick.__init__ always loads a stock scene file; load the compiled scene instead.
            go1_base.Go1Env.__init__(self, xml_path=scene_xml.write_scene_xml(scene).as_posix(), config=config)
            self._post_init()

    def _post_init(self) -> None:
        super()._post_init()
        # Joystick slices qpos[7:] for the joints; free scene objects come after them.
        self._default_pose = self._init_q[7:ROBOT_NQ]
        self._lowers, self._uppers = self.mj_model.jnt_range[1:ROBOT_NQ - 6].T
        self._soft_lowers = self._lowers * self._config.soft_joint_pos_limit_factor
        self._soft_uppers = self._uppers * self._config.soft_joint_pos_limit_factor
        self._placed = tuple(self._locate(o) for o in self._scene.objects)

    def _locate(self, obj: SceneObject) -> _Placed:
        body = self.mj_model.body(scene_xml.body_name(obj))
        mocap_id = int(self.mj_model.body_mocapid[body.id])
        qpos_adr = dof_adr = -1
        if obj.movable:
            joint = self.mj_model.joint(scene_xml.body_name(obj))
            qpos_adr, dof_adr = int(self.mj_model.jnt_qposadr[joint.id]), int(self.mj_model.jnt_dofadr[joint.id])
        return _Placed(obj, body.id, mocap_id, qpos_adr, dof_adr)

    @property
    def term_names(self) -> tuple[str, ...]:
        return self._term_names

    @property
    def scene(self) -> SceneSpec:
        return self._scene

    def features(self, data: mjx.Data, info: dict[str, Any], contact: jax.Array) -> dict[str, jax.Array]:
        feet_z = data.site_xpos[self._feet_site_id][:, 2]
        return {
            "base_lin_vel": self.get_local_linvel(data),
            "base_ang_vel": self.get_gyro(data),
            "gravity": self.get_gravity(data),
            "up_z": self.get_upvector(data)[2],
            "base_height": data.qpos[2],
            "joint_pos": data.qpos[7:ROBOT_NQ] - self._default_pose,
            "joint_vel": data.qvel[6:ROBOT_NV],
            "foot_contact": contact.astype(jnp.float32),
            "feet_air_time": info["feet_air_time"],
            "feet_height": feet_z,
            "torques": data.actuator_force,
            "base_pos": data.qpos[0:3],
            "base_yaw": self._yaw(data),
            "time": jnp.asarray(data.time, dtype=jnp.float32),
            **self.scene_features(data),
        }

    def _yaw(self, data: mjx.Data) -> jax.Array:
        forward = data.xmat[self._torso_body_id][:, 0]
        return jnp.arctan2(forward[1], forward[0])

    def scene_features(self, data: mjx.Data) -> dict[str, jax.Array]:
        """Per-object features (see go1_spec.OBJECT_FEATURES), plus target_* aliases and,
        with a goal, where the target sits relative to it (go1_spec.GOAL_FEATURES)."""
        base = data.qpos[0:3]
        yaw = self._yaw(data)
        c, s = jnp.cos(yaw), jnp.sin(yaw)
        out: dict[str, jax.Array] = {}
        for p in self._placed:
            pos = data.xpos[p.body_id]
            d = pos - base
            rel = jnp.stack([c * d[0] + s * d[1], -s * d[0] + c * d[1], d[2]])
            values = {
                "rel_pos": rel,
                "dist": jnp.sqrt(jnp.square(rel[0]) + jnp.square(rel[1])),
                "heading_err": jnp.arctan2(rel[1], rel[0]),
                "pos": pos,
            }
            if p.obj.movable:
                values["vel"] = data.qvel[p.dof_adr:p.dof_adr + 3]
            for suffix, value in values.items():
                out[f"{p.obj.name}_{suffix}"] = value
                if p.obj.name == self._scene.target:
                    out[f"target_{suffix}"] = value
        goal = self._scene.goal
        if goal is not None:
            d = out[f"{goal}_pos"] - out["target_pos"]
            out["target_to_goal_rel"] = jnp.stack([c * d[0] + s * d[1], -s * d[0] + c * d[1], d[2]])
            out["target_to_goal_dist"] = jnp.sqrt(jnp.square(d[0]) + jnp.square(d[1]))
            # How fast the gap closes: minus d/dt of target_to_goal_dist (the goal never moves).
            out["target_to_goal_speed"] = jnp.dot(out["target_vel"][:2], d[:2]) / jnp.maximum(
                out["target_to_goal_dist"], 1e-3)
        return out

    def _scene_obs(self, data: mjx.Data) -> jax.Array:
        """What the policy sees of the scene: per object its relative position, distance and
        bearing (as cos, sin), its velocity if movable, then the episode's elapsed fraction."""
        feats = self.scene_features(data)
        yaw = self._yaw(data)
        c, s = jnp.cos(yaw), jnp.sin(yaw)
        parts = []
        for p in self._placed:
            n = p.obj.name
            rel = jnp.clip(feats[f"{n}_rel_pos"][:2], -MAX_OBS_DIST, MAX_OBS_DIST)
            he = feats[f"{n}_heading_err"]
            parts += [rel, jnp.minimum(feats[f"{n}_dist"], MAX_OBS_DIST)[None], jnp.cos(he)[None], jnp.sin(he)[None]]
            if p.obj.movable:
                v = feats[f"{n}_vel"]
                parts.append(jnp.stack([c * v[0] + s * v[1], -s * v[0] + c * v[1]]))
        episode_s = self._config.episode_length * self.dt
        parts.append(jnp.asarray(data.time / episode_s, dtype=jnp.float32)[None])
        return jnp.concatenate(parts)

    def _get_obs(self, data: mjx.Data, info: dict[str, Any]) -> dict[str, jax.Array]:
        if self._scene.is_empty:
            return super()._get_obs(data, info)
        # Joystick reads the joints as qpos[7:] and qvel[6:]; hand it the robot's slice only.
        robot = data.replace(qpos=data.qpos[:ROBOT_NQ], qvel=data.qvel[:ROBOT_NV])
        obs = super()._get_obs(robot, info)
        extra = self._scene_obs(data)
        return {k: jnp.concatenate([v, extra]) for k, v in obs.items()}

    def reset(self, rng: jax.Array) -> mjx_env.State:
        state = super().reset(rng) if self._scene.is_empty else self._reset_scene(rng)
        if self._behavior_metrics:
            state.metrics.update(self.behavior(state.data))
        return state

    def _reset_scene(self, rng: jax.Array) -> mjx_env.State:
        """Joystick.reset, but with qpos/ctrl sized for the scene and the objects placed.

        Joystick.reset sets ctrl to qpos[7:], which is too long once movable objects
        add qpos, so the robot part is redone here, keeping its randomization.
        """
        qpos = self._init_q
        qvel = jnp.zeros(self.mjx_model.nv)

        # Same start as Joystick: x, y += U(-0.5, 0.5), yaw = U(-3.14, 3.14), d(xyzrpy) = U(-0.5, 0.5).
        rng, key = jax.random.split(rng)
        qpos = qpos.at[0:2].set(qpos[0:2] + jax.random.uniform(key, (2,), minval=-0.5, maxval=0.5))
        rng, key = jax.random.split(rng)
        yaw = jax.random.uniform(key, (1,), minval=-3.14, maxval=3.14)
        quat = mjx_math.axis_angle_to_quat(jnp.array([0, 0, 1]), yaw)
        qpos = qpos.at[3:7].set(mjx_math.quat_mul(qpos[3:7], quat))
        rng, key = jax.random.split(rng)
        qvel = qvel.at[0:6].set(jax.random.uniform(key, (6,), minval=-0.5, maxval=0.5))

        rng, key = jax.random.split(rng)
        qpos, mocap_pos, mocap_quat = self._place_objects(key, qpos, yaw[0])

        data = mjx_env.make_data(
            self.mj_model,
            qpos=qpos,
            qvel=qvel,
            ctrl=qpos[7:ROBOT_NQ],
            # make_data reshapes these by nmocap, which fails with no fixed objects (rough ground alone).
            **({"mocap_pos": mocap_pos, "mocap_quat": mocap_quat} if self.mj_model.nmocap else {}),
            impl=self.mjx_model.impl.value,
            naconmax=self._config.naconmax,
            njmax=self._config.njmax,
        )
        data = mjx.forward(self.mjx_model, data)

        info = self._initial_info(rng)
        metrics = {f"reward/{k}": jnp.zeros(()) for k in self._config.reward_config.scales.keys()}
        metrics["swing_peak"] = jnp.zeros(())
        obs = self._get_obs(data, info)
        reward, done = jnp.zeros(2)
        return mjx_env.State(data, obs, reward, done, metrics, info)

    def _place_objects(self, rng: jax.Array, qpos: jax.Array, robot_yaw: jax.Array):
        """Draw each object's distance, bearing and turn from its ranges, relative to the robot's start."""
        mocap_pos = jnp.asarray(self.mj_model.body_pos[self.mj_model.body_mocapid >= 0], dtype=jnp.float32)
        mocap_quat = jnp.asarray(self.mj_model.body_quat[self.mj_model.body_mocapid >= 0], dtype=jnp.float32)
        # body_pos rows above are in body order; mocap ids are too, since both follow the body tree.
        for p in self._placed:
            o = p.obj
            rng, kd, kb, ky = jax.random.split(rng, 4)
            dist = jax.random.uniform(kd, minval=o.distance[0], maxval=o.distance[1])
            bearing = jnp.deg2rad(jax.random.uniform(kb, minval=o.bearing[0], maxval=o.bearing[1]))
            turn = robot_yaw + jnp.deg2rad(jax.random.uniform(ky, minval=o.yaw[0], maxval=o.yaw[1]))
            heading = robot_yaw + bearing
            pos = jnp.stack([
                qpos[0] + dist * jnp.cos(heading),
                qpos[1] + dist * jnp.sin(heading),
                jnp.float32(scene_xml.spawn_z(o, self._scene.terrain)),
            ])
            quat = jnp.stack([jnp.cos(turn / 2), 0.0, 0.0, jnp.sin(turn / 2)])
            if p.mocap_id >= 0:
                mocap_pos = mocap_pos.at[p.mocap_id].set(pos)
                mocap_quat = mocap_quat.at[p.mocap_id].set(quat)
            else:
                qpos = qpos.at[p.qpos_adr:p.qpos_adr + 7].set(jnp.concatenate([pos, quat]))
        return qpos, mocap_pos, mocap_quat

    def _initial_info(self, rng: jax.Array) -> dict[str, Any]:
        """The info dict Joystick.reset builds (perturbation and command state included)."""
        cfg = self._config
        rng, key1, key2, key3 = jax.random.split(rng, 4)
        pert_wait = jax.random.uniform(key1, minval=cfg.pert_config.kick_wait_times[0],
                                       maxval=cfg.pert_config.kick_wait_times[1])
        pert_duration_s = jax.random.uniform(key2, minval=cfg.pert_config.kick_durations[0],
                                             maxval=cfg.pert_config.kick_durations[1])
        pert_mag = jax.random.uniform(key3, minval=cfg.pert_config.velocity_kick[0],
                                      maxval=cfg.pert_config.velocity_kick[1])
        rng, key1, key2 = jax.random.split(rng, 3)
        time_until_next_cmd = jax.random.exponential(key1) * 5.0
        cmd = jax.random.uniform(key2, shape=(3,), minval=-self._cmd_a, maxval=self._cmd_a)
        return {
            "rng": rng,
            "command": cmd,
            "steps_until_next_cmd": jnp.round(time_until_next_cmd / self.dt).astype(jnp.int32),
            "last_act": jnp.zeros(self.mjx_model.nu),
            "last_last_act": jnp.zeros(self.mjx_model.nu),
            "feet_air_time": jnp.zeros(4),
            "last_contact": jnp.zeros(4, dtype=bool),
            "swing_peak": jnp.zeros(4),
            "steps_until_next_pert": jnp.round(pert_wait / self.dt).astype(jnp.int32),
            "pert_duration_seconds": pert_duration_s,
            "pert_duration": jnp.round(pert_duration_s / self.dt).astype(jnp.int32),
            "steps_since_last_pert": 0,
            "pert_steps": 0,
            "pert_dir": jnp.zeros(3),
            "pert_mag": pert_mag,
        }


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
        out = {
            "behavior/forward_velocity": lin[0],
            "behavior/lateral_velocity": lin[1],
            "behavior/yaw_rate": self.get_gyro(data)[2],
            "behavior/base_height": data.qpos[2],
            "behavior/upright": self.get_upvector(data)[2],
        }
        if self._scene.target is not None:  # how the task is going, for ranking scene runs mid-training
            feats = self.scene_features(data)
            out["behavior/target_dist"] = feats["target_dist"]
            if self._scene.goal is not None:
                out["behavior/target_to_goal_dist"] = feats["target_to_goal_dist"]
        return out

    def _get_reward(self, data, action, info, metrics, done, first_contact, contact):
        del metrics, done, first_contact
        terms = self._reward_fn(self.features(data, info, contact), action, info["last_act"])
        return {k: jnp.asarray(terms[k], dtype=jnp.float32).reshape(()) for k in self._term_names}


def dummy_state(scene: SceneSpec = EMPTY) -> dict[str, jax.Array]:
    """A zero-filled state with the real keys and shapes, for checking a reward eagerly.
    Objects sit 1 m straight ahead, so distances are not zero."""
    state = {f.name: jnp.zeros(f.shape, dtype=jnp.float32) for f in features_for(scene)}
    state["gravity"] = jnp.array([0.0, 0.0, -1.0])
    state["up_z"] = jnp.float32(1.0)
    for name in [o.name for o in scene.objects] + (["target"] if scene.target else []):
        state[f"{name}_rel_pos"] = jnp.array([1.0, 0.0, 0.0])
        state[f"{name}_pos"] = jnp.array([1.0, 0.0, 0.0])
        state[f"{name}_dist"] = jnp.float32(1.0)
    if scene.goal is not None:
        state["target_to_goal_rel"] = jnp.array([1.0, 0.0, 0.0])
        state["target_to_goal_dist"] = jnp.float32(1.0)
        state["target_to_goal_speed"] = jnp.float32(0.0)
    return state


def discover_terms(reward_fn: RewardFn, scene: SceneSpec = EMPTY) -> tuple[str, ...]:
    """Call the reward once on a dummy state and return its term names, sorted."""
    zeros = jnp.zeros(12, dtype=jnp.float32)
    try:
        terms = reward_fn(dummy_state(scene), zeros, zeros)
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


def make_config(
    term_names: tuple[str, ...],
    impl: str = "jax",
    overrides: Mapping[str, Any] | None = None,
    scene: SceneSpec = EMPTY,
):
    config = registry.get_default_config(BASE_ENV)
    # Must happen before the env locks its config: scales are looked up by term name.
    config.reward_config.scales = config_dict.ConfigDict({k: 1.0 for k in term_names})
    config.command_config.a = [0.0, 0.0, 0.0]  # no joystick command
    config.impl = impl
    if not scene.is_empty:
        config.naconmax, config.njmax = scene_xml.contact_budget(scene)
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
    source: str,
    impl: str | None = None,
    overrides: Mapping[str, Any] | None = None,
    scene: SceneSpec | None = None,
    behavior_metrics: bool = False,
) -> Go1Env:
    """Build a Go1 env from reward source code. Raises RewardError for bad rewards."""
    impl = impl or default_impl()
    scene = scene or EMPTY
    try:
        reward_fn = load_reward(source)
    except Exception as exc:
        raise RewardError(f"{type(exc).__name__}: {exc}") from exc
    term_names = discover_terms(reward_fn, scene)
    mjx_env.ensure_menagerie_exists()
    return Go1Env(reward_fn, term_names, make_config(term_names, impl, overrides, scene), scene, behavior_metrics)
