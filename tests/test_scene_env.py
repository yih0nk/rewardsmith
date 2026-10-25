"""Structures, launched objects and terrain features in the Go1 env, and the scene gate."""

import numpy as np
import pytest

pytest.importorskip("mujoco_playground")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from rewardsmith.sim import go1, go1_spec, scene, scene_gate  # noqa: E402
from rewardsmith.sim.run_one import WARM_UP_REWARD  # noqa: E402

STAIRS = {"name": "stairs", "kind": "stairs", "color": "gray", "distance": 1.0, "bearing": 0}
RAMP = {"name": "ramp", "kind": "ramp", "color": "orange", "distance": 1.0, "bearing": 90,
        "params": {"angle": 10, "length": 2}}
ROLLED = {"name": "ball", "kind": "ball", "color": "red", "size": [0.12], "distance": [4, 5], "bearing": [-30, 30],
          "movable": True, "mass": 0.5, "speed": [2, 3]}


@pytest.fixture(scope="module")
def stairs_env():
    return go1.make_env(WARM_UP_REWARD, impl="jax", scene=scene.parse({"structures": [STAIRS, RAMP]}))


def test_ground_height_follows_steps_and_ramps(stairs_env):
    d = jax.jit(stairs_env.reset)(jax.random.PRNGKey(1)).data
    base, yaw = np.asarray(d.qpos[:2]), float(stairs_env._yaw(d))

    def along(angle, dists):
        return jnp.array([base + r * np.array([np.cos(yaw + angle), np.sin(yaw + angle)]) for r in dists])

    # up four steps, across the landing, and down the mirrored steps on the far side
    stairs = np.asarray(stairs_env.ground_height(d, along(0.0, [0.9, 1.1, 1.4, 1.7, 2.0, 2.9, 3.75, 4.5])))
    np.testing.assert_allclose(stairs, [0, 0.06, 0.12, 0.18, 0.24, 0.24, 0.18, 0], atol=1e-4)
    slope = np.tan(np.radians(10))
    ramp = np.asarray(stairs_env.ground_height(d, along(np.pi / 2, [0.9, 2.0, 3.0, 3.5, 5.0, 6.5])))
    np.testing.assert_allclose(ramp, [0, slope, 2 * slope, 0.3527, slope, 0], atol=2e-3)


def test_features_match_the_spec_and_obs_appends_after_the_joystick(stairs_env):
    state = jax.jit(stairs_env.reset)(jax.random.PRNGKey(0))
    feats = stairs_env.features(state.data, state.info, state.info["last_contact"])
    want = {f.name: f.shape for f in go1_spec.features_for(stairs_env.scene)}
    assert {k: tuple(np.shape(v)) for k, v in feats.items()} == want
    stock = go1.make_env(WARM_UP_REWARD, impl="jax")
    plain = jax.jit(stock.reset)(jax.random.PRNGKey(0)).obs["state"].shape[-1]
    # two structures x (rel xy, dist, cos, sin, height) + 5 ground samples + clearance
    assert state.obs["state"].shape[-1] == plain + 2 * 6 + 6


def test_a_launched_ball_keeps_rolling_at_the_robot():
    env = go1.make_env(WARM_UP_REWARD, impl="jax", scene=scene.parse({"objects": [ROLLED]}))
    state = jax.jit(env.reset)(jax.random.PRNGKey(0))
    step = jax.jit(env.step)
    start = float(env.features(state.data, state.info, state.info["last_contact"])["ball_dist"])
    v0 = np.linalg.norm(np.asarray(state.data.qvel[18:20]))
    for _ in range(25):
        state = step(state, jnp.zeros(12))
    feats = env.features(state.data, state.info, state.info["last_contact"])
    assert 2.0 <= v0 <= 3.0
    assert np.linalg.norm(np.asarray(feats["ball_vel"][:2])) == pytest.approx(v0, rel=0.05)
    assert float(feats["ball_dist"]) < start - 0.8  # half a second at 2-3 m/s, toward the robot


def test_scene_gate_passes_good_scenes_and_names_overlaps():
    assert scene_gate.check_scene(scene.parse({"structures": [STAIRS]})) is None
    poles = [{"name": f"pole{i}", "kind": "capsule", "color": "red", "size": [0.1, 1.0], "distance": [1.5, 2.0],
              "bearing": [-5, 5]} for i in range(2)]
    error = scene_gate.check_scene(scene.parse({"objects": poles}))
    assert error and "pole0: overlaps pole1" in error

