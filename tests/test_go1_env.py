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
