"""Franka Panda environment whose reward is a generated module.

Built on Playground's PandaPickCube: same arm, block, physics and gains. We
replace its reward with the generated terms, load our own copy of its scene
(the block's color from the scene spec, two finger-block contact sensors), and
draw a target each episode: above the table for lift tasks, on it for push and
place tasks (panda_spec.target_mode).

In steer mode (skill "ee", the arm's analogue of the Go1 walker) the policy
doesn't move the joints: each step it moves a gripper target (dx, dy, dz) and
opens or closes the fingers, and Playground's analytic Panda IK
(PandaPickCubeCartesian's controller) turns that into joint targets with the
gripper pointing down. Exploration then happens in gripper space, where every
random action is a move over the table. There is no pretrained checkpoint: the
IK is exact, so the skill needs no training.
"""

from collections.abc import Callable, Mapping
from typing import Any

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from ml_collections import config_dict
from mujoco import mjx
from mujoco_playground._src import mjx_env
from mujoco_playground._src.manipulation.franka_emika_panda import panda as panda_base
from mujoco_playground._src.manipulation.franka_emika_panda import panda_kinematics
from mujoco_playground._src.manipulation.franka_emika_panda import pick

from . import warp_quiet  # noqa: F401  (silences warp's per-step solver warning)
from .bodies.base import RewardError
from .go1 import default_impl
from .panda_spec import (
    ARM_METRICS, AT_TARGET_M, CTRL_DT, CUBE_RANGE, CUBE_REST_Z, CUBE_SIZE, CUBE_X, CUBE_Y, EE_BOUNDS, EE_STEP,
    EPISODE_STEPS, FEATURES, LIFT_HEIGHT, TRAIN_EPISODE_STEPS, target_mode,
)
from .reward_slot import load_reward
from .scene import COLORS, EMPTY, SceneSpec

BASE_ENV = "PandaPickCube"
RewardFn = Callable[[dict[str, jax.Array], jax.Array, jax.Array], Mapping[str, Any]]

# The robot's slice of qpos/qvel: 7 arm joints + 2 fingers. The block's free joint comes after.
ROBOT_NQ, ROBOT_NV = 9, 9
EE_ACTION = 4  # dx, dy, dz, grip
JOINT_LIMIT_SLACK = 0.05  # rad past a joint limit that ends the episode
HOME_QPOS = "0 0.3 0 -1.57079 0 2.0 -0.7853 0.04 0.04"
HOME_CTRL = "0 0.3 0 -1.57079 0 2.0 -0.7853 0.04"

SCENE_XML = """<mujoco model="rewardsmith panda">
  <include file="mjx_scene.xml"/>
  <worldbody>
    <body name="box" pos="{x} 0 {z}">
      <freejoint/>
      <geom type="box" name="box" size="{size}" condim="3" friction="1 .03 .003" rgba="{rgb} 1"
       contype="2" conaffinity="1" solref="0.01 1"/>
    </body>
    <body mocap="true" name="mocap_target">
      <geom type="box" size="{size}" rgba="{rgb} 0.25" contype="0" conaffinity="0"/>
    </body>
  </worldbody>
  <sensor>
    <contact name="left_finger_pad_floor_found" geom1="left_finger_pad" geom2="floor" reduce="mindist" num="1" data="found"/>
    <contact name="right_finger_pad_floor_found" geom1="right_finger_pad" geom2="floor" reduce="mindist" num="1" data="found"/>
    <contact name="hand_capsule_floor_found" geom1="hand_capsule" geom2="floor" reduce="mindist" num="1" data="found"/>
    <contact name="box_hand_found" geom1="box" geom2="hand_capsule" reduce="mindist" num="1" data="found"/>
    <contact name="left_pad_box_found" geom1="left_finger_pad" geom2="box" reduce="mindist" num="1" data="found"/>
    <contact name="right_pad_box_found" geom1="right_finger_pad" geom2="box" reduce="mindist" num="1" data="found"/>
  </sensor>
  <keyframe>
    <key name="home" qpos="{home_qpos} {x} 0 {z} 1 0 0 0" ctrl="{home_ctrl}"/>
  </keyframe>
</mujoco>
"""


def scene_xml(scene: SceneSpec = EMPTY) -> str:
    target = scene.target_object
    rgb = COLORS[target.color] if target is not None else (0.18, 0.70, 0.25)
    return SCENE_XML.format(
        x=CUBE_X, z=CUBE_REST_Z, size=" ".join(f"{s:g}" for s in CUBE_SIZE), rgb=" ".join(f"{c:g}" for c in rgb),
        home_qpos=HOME_QPOS, home_ctrl=HOME_CTRL,
    )


class PandaEnv(panda_base.PandaBase):
    def __init__(
        self,
        reward_fn: RewardFn,
        term_names: tuple[str, ...],
        config: config_dict.ConfigDict,
        scene: SceneSpec = EMPTY,
        behavior_metrics: bool = False,
        ee: bool = True,
    ):
        mjx_env.MjxEnv.__init__(self, config)  # PandaBase.__init__ reads a file; we build from a string
        self._reward_fn = reward_fn
        self._term_names = term_names
        self._scene = scene
        self._behavior_metrics = behavior_metrics
        self._ee = ee
        self._target_mode = target_mode(scene)
        self._xml_path = "rewardsmith_panda.xml"
        self._model_assets = panda_base.get_assets()
        mj_model = mujoco.MjModel.from_xml_string(scene_xml(scene), assets=self._model_assets)
        mj_model.opt.timestep = config.sim_dt
        self._mj_model = mj_model
        self._mjx_model = mjx.put_model(mj_model, impl=config.impl)
        self._action_scale = config.action_scale
        self._post_init(obj_name="box", keyframe="home")
        self._start_tip = panda_kinematics.compute_franka_fk(jnp.asarray(self._init_ctrl[:7]))
        sensor = lambda name: int(mj_model.sensor_adr[mj_model.sensor(name).id])  # noqa: E731
        self._pad_box = np.array([sensor("left_pad_box_found"), sensor("right_pad_box_found")])
        self._hand_floor = np.array([sensor(f"{g}_floor_found") for g in ("left_finger_pad", "right_finger_pad", "hand_capsule")])
        self._arm_lowers, self._arm_uppers = mj_model.jnt_range[:7].T
        self._finger_qposadr = self._robot_qposadr[7:]

    @property
    def action_size(self) -> int:
        return EE_ACTION if self._ee else self.mjx_model.nu

    @property
    def steers(self) -> bool:
        return self._ee

    @property
    def term_names(self) -> tuple[str, ...]:
        return self._term_names

    @property
    def scene(self) -> SceneSpec:
        return self._scene

    # --- state the reward reads -------------------------------------------------------------

    def features(self, data: mjx.Data, info: dict[str, Any]) -> dict[str, jax.Array]:
        gripper = data.site_xpos[self._gripper_site]
        cube = data.xpos[self._obj_body]
        target = info["target_pos"]
        contact = (data.sensordata[self._pad_box] > 0).astype(jnp.float32)
        to_target = target - cube
        to_gripper = gripper - cube
        cube_vel = data.qvel[ROBOT_NV:ROBOT_NV + 3]
        return {
            "gripper_pos": gripper,
            "gripper_vel": (gripper - info["prev_gripper"]) / CTRL_DT,
            "gripper_opening": jnp.sum(data.qpos[self._finger_qposadr]),
            "finger_contact": contact,
            "grasped": contact[0] * contact[1],
            "hand_floor_contact": jnp.any(data.sensordata[self._hand_floor] > 0).astype(jnp.float32),
            "cube_pos": cube,
            "cube_vel": cube_vel,
            "cube_height": cube[2] - CUBE_REST_Z,
            "cube_up_z": data.xmat[self._obj_body][2, 2],
            "cube_to_gripper": to_gripper,
            "gripper_to_cube_dist": jnp.linalg.norm(to_gripper),
            "target_pos": target,
            "cube_to_target": to_target,
            "cube_to_target_dist": jnp.linalg.norm(to_target),
            "joint_pos": data.qpos[self._robot_arm_qposadr],
            "joint_vel": data.qvel[:7],
            "time": jnp.asarray(data.time, dtype=jnp.float32),
        }

    def _get_obs(self, data: mjx.Data, info: dict[str, Any]) -> jax.Array:
        f = self.features(data, info)
        return jnp.concatenate([
            data.qpos[:ROBOT_NQ],
            data.qvel[:ROBOT_NV],
            f["gripper_pos"],
            f["cube_pos"],
            f["cube_to_gripper"],
            f["cube_to_target"],
            data.xmat[self._obj_body].ravel()[:6],
            jnp.clip(f["cube_vel"], -5.0, 5.0),
            info["tip_target"] - f["gripper_pos"],
            f["finger_contact"],
            f["gripper_opening"][None],
        ])

    # --- reset / step -----------------------------------------------------------------------

    def reset(self, rng: jax.Array) -> mjx_env.State:
        rng, k_box, k_dist, k_dir, k_h = jax.random.split(rng, 5)
        offset = jax.random.uniform(k_box, (2,), minval=-CUBE_RANGE, maxval=CUBE_RANGE)
        box_pos = jnp.array([CUBE_X, CUBE_Y, CUBE_REST_Z]).at[:2].add(offset)
        if self._target_mode == "table":
            dist = jax.random.uniform(k_dist, minval=0.10, maxval=0.25)
            angle = jax.random.uniform(k_dir, minval=-jnp.pi, maxval=jnp.pi)
            target = box_pos + jnp.array([dist * jnp.cos(angle), dist * jnp.sin(angle), 0.0])
            lo, hi = jnp.array([EE_BOUNDS[0][0], EE_BOUNDS[1][0]]), jnp.array([EE_BOUNDS[0][1], EE_BOUNDS[1][1]])
            target = target.at[:2].set(jnp.clip(target[:2], lo + 0.03, hi - 0.03))
        else:
            xy = jax.random.uniform(k_dist, (2,), minval=-0.05, maxval=0.05)
            target = box_pos + jnp.array([xy[0], xy[1], jax.random.uniform(k_h, minval=0.15, maxval=0.30)])

        qpos = jnp.asarray(self._init_q).at[self._obj_qposadr:self._obj_qposadr + 3].set(box_pos)
        data = mjx_env.make_data(
            self._mj_model, qpos=qpos, qvel=jnp.zeros(self._mjx_model.nv), ctrl=self._init_ctrl,
            impl=self._mjx_model.impl.value, naconmax=self._config.naconmax, naccdmax=self._config.naccdmax,
            njmax=self._config.njmax,
        )
        data = data.replace(
            mocap_pos=data.mocap_pos.at[self._mocap_target, :].set(target),
            mocap_quat=data.mocap_quat.at[self._mocap_target, :].set(jnp.array([1.0, 0.0, 0.0, 0.0])),
        )
        data = mjx.forward(self._mjx_model, data)
        info = {
            "rng": rng,
            "target_pos": target,
            "tip_target": self._start_tip[:3, 3],
            "prev_gripper": data.site_xpos[self._gripper_site],
            "last_act": jnp.zeros(self.action_size),
        }
        metrics = {f"reward/{k}": jnp.zeros(()) for k in self._term_names}
        metrics["out_of_bounds"] = jnp.zeros(())
        if self._behavior_metrics:
            metrics.update(self.behavior(data, info))
        obs = self._get_obs(data, info)
        reward, done = jnp.zeros(2)
        return mjx_env.State(data, obs, reward, done, metrics, info)

    def _ee_ctrl(self, data: mjx.Data, info: dict[str, Any], action: jax.Array) -> tuple[jax.Array, jax.Array]:
        """PandaPickCubeCartesian's controller, in 3D: move the tip target, solve IK, step the fingers."""
        a = jnp.clip(action, -1.0, 1.0)
        lo = jnp.array([b[0] for b in EE_BOUNDS])
        hi = jnp.array([b[1] for b in EE_BOUNDS])
        tip = jnp.clip(info["tip_target"] + a[:3] * EE_STEP, lo, hi)
        pose = jnp.eye(4).at[:3, :3].set(self._start_tip[:3, :3]).at[:3, 3].set(tip)
        ctrl = jnp.asarray(data.ctrl)
        q = panda_kinematics.compute_franka_ik(pose, ctrl[6], ctrl[:7])
        no_soln = jnp.any(jnp.isnan(q))
        q = jnp.where(no_soln, ctrl[:7], q)
        tip = jnp.where(no_soln, info["tip_target"], tip)
        grip = jnp.where(a[3] < 0, -1.0, 1.0) * 0.02  # up to 2 cm of finger travel per step
        ctrl = ctrl.at[:7].set(q).at[7].add(grip)
        return ctrl, tip

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        info = dict(state.info)
        info["prev_gripper"] = state.data.site_xpos[self._gripper_site]
        if self._ee:
            ctrl, info["tip_target"] = self._ee_ctrl(state.data, info, action)
        else:
            ctrl = jnp.asarray(state.data.ctrl) + action * self._action_scale
        ctrl = jnp.clip(ctrl, self._lowers, self._uppers)
        data = mjx_env.step(self._mjx_model, state.data, ctrl, self.n_substeps)

        terms = self._reward_fn(self.features(data, info), action, info["last_act"])
        terms = {k: jnp.asarray(terms[k], dtype=jnp.float32).reshape(()) for k in self._term_names}
        reward = jnp.clip(sum(terms.values()), -1e4, 1e4)
        reward = jnp.where(jnp.isfinite(reward), reward, 0.0)
        info["last_act"] = action

        fell = self._fell(data)
        done = fell | jnp.isnan(data.qpos).any() | jnp.isnan(data.qvel).any()
        metrics = dict(state.metrics)
        metrics.update({f"reward/{k}": v for k, v in terms.items()})
        metrics["out_of_bounds"] = fell.astype(jnp.float32)
        if self._behavior_metrics:
            metrics.update(self.behavior(data, info))
        obs = self._get_obs(data, info)
        return mjx_env.State(data, obs, reward, done.astype(jnp.float32), metrics, info)

    def _fell(self, data: mjx.Data) -> jax.Array:
        """The arm's "fell": the block knocked off the table, or an arm joint past its limit."""
        cube = data.xpos[self._obj_body]
        off_table = jnp.any(jnp.abs(cube) > 1.0) | (cube[2] < 0.0)
        q = data.qpos[self._robot_arm_qposadr]
        past_limit = jnp.any((q < self._arm_lowers - JOINT_LIMIT_SLACK) | (q > self._arm_uppers + JOINT_LIMIT_SLACK))
        return off_table | past_limit

    def behavior(self, data: mjx.Data, info: dict[str, Any]) -> dict[str, jax.Array]:
        """How the task is going, reward aside, for ranking candidates mid-training (see Go1Env.behavior)."""
        f = self.features(data, info)
        return {
            "behavior/cube_height": f["cube_height"],
            "behavior/grasped": f["grasped"],
            "behavior/gripper_to_cube_dist": f["gripper_to_cube_dist"],
            "behavior/target_dist": f["cube_to_target_dist"],
        }

    def evaluate_rollouts(self, result: Any, out_dir: Any, steps: int = EPISODE_STEPS, body: str | None = None):
        """evaluation.evaluate's arm path (module-level evaluate_rollouts)."""
        return evaluate_rollouts(self, result, out_dir, steps, body)

    def frame(self, data: mjx.Data, info: dict[str, Any]) -> dict[str, jax.Array]:
        """What rollout_metrics needs from one step."""
        f = self.features(data, info)
        return {"cube_pos": f["cube_pos"], "target_pos": f["target_pos"], "grasped": f["grasped"],
                "cube_height": f["cube_height"], "fell": self._fell(data).astype(jnp.float32)}


EVAL_EPISODES = 32  # fresh episodes the arm metrics are measured over (the Go1's scene_eval.TARGET_ENVS)


def evaluate_rollouts(env: "PandaEnv", result: Any, out_dir: Any, steps: int = EPISODE_STEPS,
                      body: str | None = None) -> tuple[Any, list[bytes], dict[str, float], dict[str, float]]:
    """evaluation.evaluate for the arm: one compiled batch of the video episode (seed 0) and
    EVAL_EPISODES fresh ones; ARM_METRICS over the fresh ones. No robustness rollouts yet.
    The policy acts deterministically: the gripper closes on the sign of a noisy action, and
    sampled actions opened it mid-lift (dropped 28-47%, 0% with the mean action, same policies).
    Returns (mp4 path, critic PNGs, metrics, {})."""
    import time

    from . import render

    start = time.perf_counter()
    policy = result.make_inference_fn(result.params, deterministic=True)

    def one(rng):
        state = env.reset(rng)

        def body_fn(carry, _):
            s, key = carry
            key, k = jax.random.split(key)
            action, _ = policy(s.obs, k)
            s = env.step(s, action)
            return (s, key), (s.data.qpos, env.frame(s.data, s.info), s.done)

        _, (qpos, frames, done) = jax.lax.scan(body_fn, (state, rng), None, length=steps)
        return state.data.qpos, state.data.mocap_pos, state.data.mocap_quat, qpos, frames, done

    keys = jnp.concatenate([jax.random.PRNGKey(0)[None], jax.random.split(jax.random.PRNGKey(1), EVAL_EPISODES)])
    qpos0, mocap_pos, mocap_quat, qpos, frames, done = jax.device_get(jax.jit(jax.vmap(one))(keys))
    metrics = rollout_metrics({k: np.asarray(v)[1:] for k, v in frames.items()})
    video_done = np.asarray(done[0])
    n = int(np.argmax(video_done)) + 1 if video_done.any() else len(video_done)
    video_qpos = np.concatenate([qpos0[0][None], np.asarray(qpos[0])[:n]])
    video, pngs = render.render_rollout(
        env, video_qpos, out_dir, mocap_pos=mocap_pos[0], mocap_quat=mocap_quat[0], body=body)
    metrics["rollout_steps"] = float(n)
    metrics["eval_s"] = round(time.perf_counter() - start, 1)
    return video, pngs, metrics, {}


def randomizer(scene: SceneSpec = EMPTY) -> Callable[..., Any]:
    """A Panda domain randomizer, one robot per env: table and block friction U(0.6, 1.2),
    block mass x U(0.7, 1.3), arm link masses x U(0.9, 1.1), arm joint damping x U(0.9, 1.1)
    and armature x U(1.0, 1.05). Only the arm (its 9 dofs) and the block are touched.
    Called as randomizer()(mjx_model, rngs) -> (batched model, in_axes)."""
    mj_model = mujoco.MjModel.from_xml_string(scene_xml(scene), assets=panda_base.get_assets())
    floor, box = mj_model.geom("floor").id, mj_model.geom("box").id
    box_body = mj_model.body("box").id
    arm_bodies = np.array([mj_model.body(f"link{i}").id for i in range(1, 8)])

    def domain_randomize(model: Any, rng: jax.Array) -> tuple[Any, Any]:
        @jax.vmap
        def rand_dynamics(rng):
            k = jax.random.split(rng, 5)
            friction = jax.random.uniform(k[0], (2,), minval=0.6, maxval=1.2)
            geom_friction = model.geom_friction.at[floor, 0].set(friction[0]).at[box, 0].set(friction[1])
            body_mass = model.body_mass.at[box_body].multiply(jax.random.uniform(k[1], minval=0.7, maxval=1.3))
            body_mass = body_mass.at[arm_bodies].multiply(
                jax.random.uniform(k[2], (len(arm_bodies),), minval=0.9, maxval=1.1))
            damping = model.dof_damping.at[:ROBOT_NV].multiply(
                jax.random.uniform(k[3], (ROBOT_NV,), minval=0.9, maxval=1.1))
            armature = model.dof_armature.at[:ROBOT_NV].multiply(
                jax.random.uniform(k[4], (ROBOT_NV,), minval=1.0, maxval=1.05))
            return geom_friction, body_mass, damping, armature

        friction, mass, damping, armature = rand_dynamics(rng)
        fields = {"geom_friction": friction, "body_mass": mass, "dof_damping": damping, "dof_armature": armature}
        in_axes = jax.tree_util.tree_map(lambda x: None, model).tree_replace({k: 0 for k in fields})
        return model.tree_replace(fields), in_axes

    return domain_randomize


def rollout_metrics(frames: Mapping[str, np.ndarray], dt: float = CTRL_DT) -> dict[str, float]:
    """ARM_METRICS from per-step frames, each shaped (episodes, steps, ...). Steps after an
    episode's first fell are ignored (it would have been reset)."""
    cube = np.asarray(frames["cube_pos"], dtype=np.float64)
    target = np.asarray(frames["target_pos"], dtype=np.float64)
    held = np.asarray(frames["grasped"]) > 0.5
    height = np.asarray(frames["cube_height"], dtype=np.float64)
    fell = np.asarray(frames["fell"]) > 0.5
    n, t = held.shape
    alive = np.cumsum(fell, axis=1) == 0
    alive[:, 1:] |= alive[:, :-1] & fell[:, 1:]  # keep the step it fell on
    last = np.maximum(alive.sum(axis=1) - 1, 0)
    rows = np.arange(n)
    dist = np.linalg.norm(cube - target, axis=-1)
    lifted_now = (height >= LIFT_HEIGHT) & held & alive
    streak = np.zeros(n)
    best = np.zeros(n)
    for i in range(t):
        streak = np.where(lifted_now[:, i], streak + 1, 0)
        best = np.maximum(best, streak)
    ever_lifted = lifted_now.any(axis=1)
    end_dist = dist[rows, last]
    end_resting = (height[rows, last] < 0.01) & ~held[rows, last]
    table_target = np.abs(target[rows, last, 2] - CUBE_REST_Z) < 0.01
    after_lift = np.cumsum(lifted_now, axis=1) > 0
    dropped = (after_lift & ~held & (height < 0.01) & (dist > AT_TARGET_M) & alive).any(axis=1)
    out = {
        "lifted": ever_lifted.mean(),
        "lift_held_s": (best * dt).mean(),
        "max_cube_height": np.where(alive, height, 0.0).max(axis=1).mean(),
        "grasped_frac": ((held & alive).sum(axis=1) / np.maximum(alive.sum(axis=1), 1)).mean(),
        "final_target_dist": end_dist.mean(),
        "min_target_dist": np.where(alive, dist, np.inf).min(axis=1).mean(),
        "at_target": (end_dist < AT_TARGET_M).mean(),
        "placed": ((end_dist < AT_TARGET_M) & end_resting & table_target).mean(),
        "dropped": dropped.mean(),
        "cube_moved": np.linalg.norm(cube[rows, last, :2] - cube[:, 0, :2], axis=-1).mean(),
        "fell": fell.any(axis=1).mean(),
    }
    assert set(out) == set(ARM_METRICS)
    return {k: float(v) for k, v in out.items()}


def dummy_state() -> dict[str, jax.Array]:
    """A zero-filled state with the real keys and shapes, for checking a reward eagerly.
    The block rests on the table 10 cm under the gripper's home position."""
    state = {f.name: jnp.zeros(f.shape, dtype=jnp.float32) for f in FEATURES}
    state["gripper_pos"] = jnp.array([0.68, 0.0, 0.38])
    state["gripper_opening"] = jnp.float32(0.08)
    state["cube_pos"] = jnp.array([0.5, 0.0, CUBE_REST_Z])
    state["cube_up_z"] = jnp.float32(1.0)
    state["cube_to_gripper"] = state["gripper_pos"] - state["cube_pos"]
    state["gripper_to_cube_dist"] = jnp.linalg.norm(state["cube_to_gripper"])
    state["target_pos"] = jnp.array([0.5, 0.0, 0.25])
    state["cube_to_target"] = state["target_pos"] - state["cube_pos"]
    state["cube_to_target_dist"] = jnp.linalg.norm(state["cube_to_target"])
    return state


def discover_terms(reward_fn: RewardFn, action_size: int = EE_ACTION) -> tuple[str, ...]:
    """Call the reward once on a dummy state and return its term names, sorted."""
    zeros = jnp.zeros(action_size, dtype=jnp.float32)
    try:
        terms = reward_fn(dummy_state(), zeros, zeros)
    except Exception as exc:
        raise RewardError(f"reward() raised on the arm at rest: {type(exc).__name__}: {exc}") from exc
    if not isinstance(terms, Mapping) or not terms:
        raise RewardError("reward() must return a non-empty dict of named terms")
    for name, value in terms.items():
        if not isinstance(name, str) or not name.isidentifier():
            raise RewardError(f"term name {name!r} must be a plain identifier like 'lift_height'")
        size = jnp.size(jnp.asarray(value))
        if size != 1:
            raise RewardError(f"term {name!r} must be a scalar, got {size} values")
    return tuple(sorted(terms))


def make_config(impl: str = "jax", overrides: Mapping[str, Any] | None = None) -> config_dict.ConfigDict:
    config = pick.default_config()
    config.ctrl_dt = CTRL_DT
    config.episode_length = TRAIN_EPISODE_STEPS
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
    ee: bool = True,
) -> PandaEnv:
    """Build a Panda env from reward source code. Raises RewardError for bad rewards.
    ee=True (steer mode) drives the end-effector skill; False drives the joints."""
    impl = impl or default_impl()
    try:
        reward_fn = load_reward(source)
    except Exception as exc:
        raise RewardError(f"{type(exc).__name__}: {exc}") from exc
    term_names = discover_terms(reward_fn, EE_ACTION if ee else 8)
    mjx_env.ensure_menagerie_exists()
    return PandaEnv(reward_fn, term_names, make_config(impl, overrides), scene or EMPTY, behavior_metrics, ee)
