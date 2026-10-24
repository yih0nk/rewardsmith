from pathlib import Path

import pytest

pytest.importorskip("mujoco_playground")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from rewardsmith.sim import go1, train  # noqa: E402
from rewardsmith.sim.playground import PlaygroundSimulator  # noqa: E402

WALK_BACKWARDS = (Path(__file__).parent.parent / "examples" / "rewards" / "walk_backwards.py").read_text()


@pytest.fixture(scope="module")
def env():
    return go1.make_env(WALK_BACKWARDS, impl="jax")


def test_randomizer_gives_each_robot_its_own_physics(env):
    model_v, in_axes = train.randomizer()(env.mjx_model, jax.random.split(jax.random.PRNGKey(0), 3))
    assert model_v.body_mass.shape == (3, env.mjx_model.nbody)
    assert in_axes.body_mass == 0 and in_axes.geom_friction == 0
    assert not jnp.allclose(model_v.body_mass[0], model_v.body_mass[1])
    friction = model_v.geom_friction[:, 0, 0]
    assert jnp.all((friction >= 0.4) & (friction <= 1.0))


def test_randomized_robots_step_through_our_env(env):
    model_v, in_axes = train.randomizer()(env.mjx_model, jax.random.split(jax.random.PRNGKey(0), 2))

    def reset_and_step(model, rng):
        nominal = env._mjx_model
        env._mjx_model = model  # what Playground's randomization wrapper does
        try:
            state = env.reset(rng)
            return env.step(state, jnp.zeros(env.action_size))
        finally:
            env._mjx_model = nominal

    state = jax.jit(jax.vmap(reset_and_step, in_axes=(in_axes, 0)))(model_v, jax.random.split(jax.random.PRNGKey(1), 2))
    assert state.reward.shape == (2,)
    assert jnp.all(jnp.isfinite(state.reward))


def test_push_overrides_turn_on_perturbations():
    config = go1.make_config(("upright",), overrides=go1.PUSH_OVERRIDES)
    assert config.pert_config.enable is True
    assert go1.make_config(("upright",)).pert_config.enable is False


def test_train_passes_the_randomizer_only_when_asked(env, monkeypatch):
    seen = []

    def fake_train(**kwargs):
        seen.append(kwargs["randomization_fn"])
        kwargs["progress_fn"](0, {"eval/episode_reward": 1.0})
        return (lambda *a, **k: None), None, None

    monkeypatch.setattr(train.ppo, "train", fake_train)
    assert train.train(env, "smoke").randomized is False
    assert train.train(env, "smoke", randomize=True).randomized is True
    assert seen[0] is None and callable(seen[1])


def test_simulator_reads_robustness_flags(monkeypatch):
    monkeypatch.delenv("REWARDSMITH_RANDOMIZE", raising=False)
    monkeypatch.delenv("REWARDSMITH_PUSH", raising=False)
    sim = PlaygroundSimulator(preset="smoke")
    assert (sim.randomize, sim.push) == (True, False)

    monkeypatch.setenv("REWARDSMITH_RANDOMIZE", "0")
    monkeypatch.setenv("REWARDSMITH_PUSH", "1")
    sim = PlaygroundSimulator(preset="smoke")
    assert (sim.randomize, sim.push) == (False, True)
    assert PlaygroundSimulator(preset="smoke", randomize=True).randomize is True
