"""Structures, launched objects and terrain features in the Go1 env, the scene gate, and the
check code measured over evaluation episodes."""

import numpy as np
import pytest

pytest.importorskip("mujoco_playground")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from rewardsmith.sim import evaluation, go1, go1_spec, render, scene, scene_gate  # noqa: E402
from rewardsmith.sim.train import TrainResult  # noqa: E402
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


def test_check_code_is_measured_over_the_evaluation_episodes(stairs_env, tmp_path, monkeypatch):
    monkeypatch.setattr(render, "render_rollout", lambda env, qpos, out_dir, **kw: (out_dir / "v.mp4", [b"png"]))
    result = TrainResult(make_inference_fn=lambda p, deterministic: (lambda obs, key: (jnp.zeros(obs["state"].shape[:-1] + (12,)), {})),
                         params=None)
    code = ("import jax.numpy as jnp\n\ndef check(traj):\n    v = traj['valid'] > 0\n"
            "    return {'climbed': jnp.max(jnp.where(v, traj['base_ground_z'], 0.0)) > 0.2,\n"
            "            'steps': jnp.sum(traj['valid'])}\n")
    spec = [{"metric": "check:climbed", "op": ">=", "value": 0.8}, {"metric": "scene_fell", "op": "<=", "value": 0.1}]
    _, _, metrics, _ = evaluation.evaluate(stairs_env, result, tmp_path, robust=False, steps=10,
                                           success_spec=spec, check_code=code)
    assert metrics["check:climbed"] == 0.0  # a standing robot never climbs
    assert metrics["check:steps"] == pytest.approx(11.0)  # the reset row plus 10 steps, none fell
    assert metrics["scene_fell"] == 0.0
