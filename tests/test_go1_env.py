from pathlib import Path

import pytest

pytest.importorskip("mujoco_playground")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from rewardsmith.sim import go1  # noqa: E402
from rewardsmith.sim.go1_spec import FEATURES, describe  # noqa: E402

WALK_BACKWARDS = (Path(__file__).parent.parent / "examples" / "rewards" / "walk_backwards.py").read_text()


@pytest.fixture(scope="module")
def env():
    return go1.make_env(WALK_BACKWARDS)


def test_terms_come_from_the_reward(env):
    assert env.term_names == ("backward", "no_sideways", "no_spin", "smooth", "upright")


def test_reset_and_step_use_generated_terms(env):
    state = jax.jit(env.reset)(jax.random.PRNGKey(0))
    assert jnp.all(state.info["command"] == 0)  # no joystick command
    step = jax.jit(env.step)
    for _ in range(3):
        state = step(state, jnp.zeros(env.action_size))
    assert jnp.isfinite(state.reward)
    assert {f"reward/{t}" for t in env.term_names} <= set(state.metrics)
    assert "reward/tracking_lin_vel" not in state.metrics


def test_features_match_the_spec(env):
    state = jax.jit(env.reset)(jax.random.PRNGKey(1))
    contact = jnp.zeros(4, dtype=bool)
    feats = env.features(state.data, state.info, contact)
    assert set(feats) == {f.name for f in FEATURES}
    for f in FEATURES:
        assert jnp.shape(feats[f.name]) == f.shape, f.name
    assert 0.2 < float(feats["base_height"]) < 0.35


def test_describe_lists_every_feature():
    text = describe()
    assert all(f'state["{f.name}"]' in text for f in FEATURES)


@pytest.mark.parametrize(
    "source, message",
    [
        ("def nope(:", "SyntaxError"),
        ("def other(): pass", "named reward"),
        ("def reward(s, a, p):\n    return {}", "non-empty dict"),
        ("import jax.numpy as jnp\ndef reward(s, a, p):\n    return {'x': s['joint_pos']}", "scalar"),
        ("def reward(s, a, p):\n    return {'x': s['missing']}", "raised"),
    ],
)
def test_bad_rewards_raise_reward_error(source, message):
    with pytest.raises(go1.RewardError, match=message):
        go1.make_env(source)


# Scenes: objects add features, observations and per-episode placement.

from rewardsmith.sim import scene as scene_mod  # noqa: E402
from rewardsmith.sim.go1_spec import features_for  # noqa: E402

WALK_TO_BALL = (Path(__file__).parent.parent / "examples" / "rewards" / "walk_to_ball.py").read_text()
SCENE = scene_mod.parse({
    "objects": [
        {"name": "ball", "kind": "ball", "color": "red", "size": [0.15], "distance": [1.5, 3.0], "bearing": [-90, 90]},
        {"name": "crate", "kind": "box", "color": "blue", "size": [0.4, 0.4, 0.3], "distance": [2.0, 2.5],
         "bearing": 180, "movable": True, "mass": 2.0},
        {"name": "goal", "kind": "zone", "color": "green", "size": [0.5], "distance": 3.0, "bearing": -90},
    ],
    "target": "ball",
})


@pytest.fixture(scope="module")
def scene_env():
    return go1.make_env(WALK_TO_BALL, scene=SCENE)


@pytest.fixture(scope="module")
def scene_states(scene_env):
    reset = jax.jit(jax.vmap(scene_env.reset))
    return reset(jax.random.split(jax.random.PRNGKey(3), 16))


def test_scene_features_match_the_spec(scene_env, scene_states):
    state = jax.tree.map(lambda x: x[0], scene_states)
    feats = scene_env.features(state.data, state.info, jnp.zeros(4, dtype=bool))
    expected = features_for(SCENE)
    assert set(feats) == {f.name for f in expected}
    for f in expected:
        assert jnp.shape(feats[f.name]) == f.shape, f.name
    assert "crate_vel" in feats and "ball_vel" not in feats  # velocity only for movable objects
    assert jnp.allclose(feats["target_dist"], feats["ball_dist"])
    assert jnp.shape(feats["joint_pos"]) == (12,)  # the crate's free joint stays out of the robot's slice


def test_objects_are_placed_afresh_within_their_ranges(scene_env, scene_states):
    feats = jax.vmap(lambda d: scene_env.scene_features(d))(scene_states.data)
    dist = feats["ball_dist"]  # from the base, which starts where the ranges are measured from
    assert jnp.all((dist > 1.5 - 0.01) & (dist < 3.0 + 0.01))
    assert jnp.all(jnp.abs(feats["ball_heading_err"]) <= jnp.pi / 2 + 0.01)
    assert jnp.all(jnp.abs(jnp.abs(feats["crate_heading_err"]) - jnp.pi) < 0.01)  # straight behind
    assert float(jnp.std(feats["ball_pos"][:, 0])) > 0.1  # a different spot each episode
    assert jnp.allclose(feats["goal_dist"], 3.0, atol=0.01)


def test_scene_adds_to_the_observation(env, scene_env, scene_states):
    plain = jax.jit(env.reset)(jax.random.PRNGKey(0))
    per_object = 5 * 3 + 2  # rel x, y, dist, cos, sin each; crate also has velocity
    assert scene_states.obs["state"].shape[-1] == plain.obs["state"].shape[-1] + per_object + 1  # + time
    assert scene_states.obs["privileged_state"].shape[-1] == plain.obs["privileged_state"].shape[-1] + per_object + 1


def test_scene_env_steps_with_finite_rewards(scene_env, scene_states):
    state = jax.tree.map(lambda x: x[0], scene_states)
    step = jax.jit(scene_env.step)
    for _ in range(3):
        state = step(state, jnp.zeros(scene_env.action_size))
    assert jnp.isfinite(state.reward)
    assert float(state.data.time) > 0
    feats = scene_env.features(state.data, state.info, jnp.zeros(4, dtype=bool))
    assert float(feats["time"]) == pytest.approx(3 * scene_env.dt, rel=1e-3)


def test_rough_terrain_scene_builds():
    rough = scene_mod.parse({"terrain": "rough"})
    env = go1.make_env(WALK_BACKWARDS, scene=rough)
    assert env.mj_model.geom("floor").type == 1  # mjGEOM_HFIELD
    assert env._config.njmax > 40


def test_reward_reading_a_missing_object_is_refused():
    source = "def reward(s, a, p):\n    return {'x': s['cube_dist']}"
    with pytest.raises(go1.RewardError, match="cube_dist"):
        go1.make_env(source, scene=SCENE)
