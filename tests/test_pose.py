from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mujoco_playground")

from rewardsmith.sim import go1, pose, scene  # noqa: E402

WALK = (Path(__file__).parent.parent / "examples" / "rewards" / "walk_forward.py").read_text()


@pytest.fixture(scope="module")
def env():
    return go1.make_env(WALK, impl="jax")


def _reared(env, pitch_deg: float) -> np.ndarray:
    """The standing pose pitched nose-up, raised so the rear feet rest on the floor as when standing."""
    import mujoco

    q = np.array(env._init_q, dtype=float)
    half = np.radians(pitch_deg) / 2
    q[3:7] = [np.cos(half), 0.0, -np.sin(half), 0.0]  # about -y: nose up
    data = mujoco.MjData(env.mj_model)
    data.qpos[:] = q
    mujoco.mj_kinematics(env.mj_model, data)
    rear_z = data.site_xpos[np.asarray(env._feet_site_id)[2:], 2].min()
    q[2] += 0.0065 - rear_z
    return q


def test_standing_has_all_feet_down_and_a_level_body(env):
    m = pose.metrics(env, np.stack([env._init_q] * 10))
    assert m["front_feet_off"] == 0 and m["rear_feet_off"] == 0 and m["feet_off_mean"] == 0
    assert abs(m["pitch_deg"]) < 1 and m["abs_roll_deg"] < 1 and m["front_off_held_s"] == 0


def test_rearing_up_lifts_the_front_feet_and_pitches_nose_up(env):
    up = _reared(env, 60)
    qpos = np.stack([env._init_q] * 5 + [up] * 10 + [env._init_q] * 3 + [up] * 2)
    m = pose.metrics(env, qpos)
    assert m["front_feet_off"] == pytest.approx(12 / 20)
    assert m["rear_feet_off"] == 0
    assert m["front_off_held_s"] == pytest.approx(10 * env.dt)
    assert m["pitch_deg"] == pytest.approx(12 / 20 * 60, abs=1)


def test_pitch_matches_the_reward_features(env):
    # The writer is told pitch = arcsin(-gravity[0]); check that sign against the env.
    import jax
    import jax.numpy as jnp

    state = jax.jit(env.reset)(jax.random.PRNGKey(0))
    q = jnp.asarray(_reared(env, 45))
    data = state.data.replace(qpos=q, qvel=jnp.zeros_like(state.data.qvel))
    data = jax.jit(lambda d: __import__("mujoco").mjx.forward(env.mjx_model, d))(data)
    gravity = env.get_gravity(data)
    assert np.degrees(np.arcsin(-float(gravity[0]))) == pytest.approx(45, abs=1)
    assert pose.metrics(env, q[None])["pitch_deg"] == pytest.approx(45, abs=1)


def test_counted_stops_at_the_first_fall():
    assert pose.counted(np.array([0, 0, 1, 1])) == 3
    assert pose.counted(np.zeros(4)) == 4


def test_rough_ground_raises_the_off_floor_bar():
    rough = go1.make_env(WALK, impl="jax", scene=scene.parse({"terrain": "rough"}))
    q = np.array(rough._init_q, dtype=float)
    q[2] = 0.278 + 0.06  # standing pose 6 cm up (feet at ~6.6 cm): on rough ground that could be a bump
    assert pose.metrics(rough, q[None])["feet_off_mean"] == 0
