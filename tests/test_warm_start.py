"""Warm starts: stock gait terms, checkpoints, and fitting a walker to a scene env."""

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mujoco_playground")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from brax.training.agents.ppo import networks as ppo_networks  # noqa: E402

from rewardsmith.sim import checkpoints, go1, train  # noqa: E402
from rewardsmith.sim import scene as scene_mod  # noqa: E402

ROOT = Path(__file__).parent.parent / "examples"
WALK_BACKWARDS = (ROOT / "rewards" / "walk_backwards.py").read_text()
WALK_TO_BALL = (ROOT / "rewards" / "walk_to_ball.py").read_text()
SCENE = scene_mod.parse((ROOT / "scenes" / "red_ball.json").read_text())


def test_stock_terms_sit_under_the_reward():
    env = go1.make_env(WALK_BACKWARDS, impl="jax", stock_profile="gait")
    scales = env._config.reward_config.scales
    assert env.term_names == ("backward", "no_sideways", "no_spin", "smooth", "upright")
    assert scales["stock_orientation"] == go1.STOCK_SCALES["orientation"] * go1.STOCK_WEIGHT
    assert "stock_tracking_lin_vel" not in scales  # it would reward standing still
    state = jax.jit(env.step)(jax.jit(env.reset)(jax.random.PRNGKey(0)), jnp.zeros(12))
    assert {"reward/stock_orientation", "reward/backward"} <= set(state.metrics)
    assert np.isfinite(float(state.reward))


def test_no_profile_leaves_the_reward_as_written():
    env = go1.make_env(WALK_BACKWARDS, impl="jax")
    assert set(env._config.reward_config.scales) == set(env.term_names)


def test_pose_profile_and_the_rewards_own_overrides():
    pose = go1.stock_scales(go1.load_reward(WALK_BACKWARDS), "pose")
    assert "orientation" not in pose and "torques" in pose
    own = WALK_BACKWARDS + "\nSTOCK_TERMS = {'orientation': 0.0, 'pose': 2.0}\n"
    gait = go1.stock_scales(go1.load_reward(own), "gait")
    assert "orientation" not in gait and gait["pose"] == 2.0 * go1.STOCK_SCALES["pose"] * go1.STOCK_WEIGHT


@pytest.mark.parametrize("extra, message", [
    ("STOCK_TERMS = {'tracking_lin_vel': 1.0}", "unknown terms"),
    ("STOCK_TERMS = ['pose']", "must be a dict"),
])
def test_bad_stock_terms_are_reward_errors(extra, message):
    with pytest.raises(go1.RewardError, match=message):
        go1.make_env(WALK_BACKWARDS + "\n" + extra + "\n", impl="jax", stock_profile="gait")


def _walker_params(env, seed=0):
    """Random 'walker' params for the plain env, with a non-trivial normalizer."""
    kwargs = train.ppo_kwargs("smoke")
    nets = kwargs["network_factory"](env.observation_size, env.action_size)
    from brax.training.acme import running_statistics, specs

    sizes = {k: specs.Array(v[-1:], jnp.float32) for k, v in env.observation_size.items()}
    norm = running_statistics.init_state(sizes)
    k1, k2, k3 = jax.random.split(jax.random.PRNGKey(seed), 3)
    norm = norm.replace(
        mean={k: jax.random.normal(k3, v.shape) for k, v in norm.mean.items()},
        std={k: 1.0 + jnp.abs(jax.random.normal(k3, v.shape)) for k, v in norm.std.items()},
        count=jnp.float32(1e6),
    )
    return nets, (norm, nets.policy_network.init(k1), nets.value_network.init(k2))


def test_a_fitted_walker_ignores_the_scene_features(tmp_path):
    plain = go1.make_env(WALK_BACKWARDS, impl="jax")
    scene_env = go1.make_env(WALK_TO_BALL, impl="jax", scene=SCENE)
    _, params = _walker_params(plain)
    path = checkpoints.save(tmp_path / "params.pkl", params)
    fitted = train.fit_params(scene_env, checkpoints.load(path))
    assert checkpoints.obs_sizes(fitted) == {k: v[-1] for k, v in scene_env.observation_size.items()}

    def act(env, params, obs):
        nets = train.ppo_kwargs("smoke")["network_factory"](env.observation_size, env.action_size)
        policy = ppo_networks.make_inference_fn(nets)(params, deterministic=True)
        return policy(obs, jax.random.PRNGKey(0))[0]

    obs = jax.jit(scene_env.reset)(jax.random.PRNGKey(1)).obs
    walker_obs = {k: v[: plain.observation_size[k][-1]] for k, v in obs.items()}
    moved = {k: v.at[-4:].add(3.0) for k, v in obs.items()}  # only scene features (they come last)
    np.testing.assert_allclose(act(scene_env, fitted, obs), act(plain, params, walker_obs), atol=1e-5)
    np.testing.assert_allclose(act(scene_env, fitted, moved), act(plain, params, walker_obs), atol=1e-5)


def test_fitting_to_a_smaller_env_is_an_error():
    plain = go1.make_env(WALK_BACKWARDS, impl="jax")
    scene_env = go1.make_env(WALK_TO_BALL, impl="jax", scene=SCENE)
    _, params = _walker_params(scene_env)
    with pytest.raises(ValueError, match="gives only"):
        train.fit_params(plain, params)


def test_latest_checkpoint(tmp_path):
    assert checkpoints.latest(tmp_path) is None
    for steps in (0, 2_000_000, 10_000_000, 400_000):
        checkpoints.save(checkpoints.step_path(tmp_path, steps), {"x": np.zeros(1)})
    assert checkpoints.latest(tmp_path).name == "params_10000000.pkl"


def _real_walker(tmp_path):
    """A random walker-shaped checkpoint (state 48 inputs), saved like train_walker.py does."""
    plain = go1.make_env(WALK_BACKWARDS, impl="jax")
    _, params = _walker_params(plain)
    return checkpoints.save(tmp_path / "params.pkl", params), params


def test_steer_mode_commands_the_walker(tmp_path):
    path, params = _real_walker(tmp_path)
    env = go1.make_env(WALK_TO_BALL, impl="jax", scene=SCENE, walker=path)
    assert env.steers and env.action_size == 3
    state = jax.jit(env.reset)(jax.random.PRNGKey(0))
    step = jax.jit(env.step)
    for _ in range(5):
        state = step(state, jnp.array([1.0, 0.0, -1.0]))
    # the command eases toward (1.5, 0, -1.2) and is what the policy observes
    expected = (1 - (1 - go1.COMMAND_SMOOTHING) ** 5) * np.array([1.5, 0.0, -1.2])
    np.testing.assert_allclose(state.info["command"], expected, rtol=1e-5)
    np.testing.assert_allclose(state.obs["state"][45:48], expected, rtol=1e-5)
    # the joints got the walker's action for that observation
    walker = go1.walker_policy(params)
    assert state.info["last_act"].shape == (12,)
    assert np.isfinite(float(state.reward))
    assert walker({"state": state.obs["state"][:48]}, jax.random.PRNGKey(0))[0].shape == (12,)


def test_steer_mode_needs_a_walker_checkpoint(tmp_path):
    scene_env = go1.make_env(WALK_TO_BALL, impl="jax", scene=SCENE)
    _, params = _walker_params(scene_env)
    with pytest.raises(ValueError, match="not a walker checkpoint"):
        go1.walker_policy(params)


def test_reset_noise_keeps_the_mean_and_sets_the_std():
    plain = go1.make_env(WALK_BACKWARDS, impl="jax")
    _, params = _walker_params(plain)
    nets = train.ppo_kwargs("smoke")["network_factory"](plain.observation_size, plain.action_size)
    obs = jax.jit(plain.reset)(jax.random.PRNGKey(2)).obs
    before = nets.policy_network.apply(params[0], params[1], obs)
    after = nets.policy_network.apply(*checkpoints.reset_noise(params, 0.5)[:2], obs)
    np.testing.assert_allclose(after[:12], before[:12], atol=1e-6)
    np.testing.assert_allclose(jax.nn.softplus(after[12:]) + checkpoints.MIN_STD, 0.5, rtol=1e-5)
