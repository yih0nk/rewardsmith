import pytest

pytest.importorskip("mujoco_playground")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from rewardsmith.sim import g1, pose  # noqa: E402
from rewardsmith.sim.bodies import get_body  # noqa: E402
from rewardsmith.sim.bodies.base import RewardError  # noqa: E402
from rewardsmith.sim.g1_spec import FEATURES, POSE_METRICS, describe  # noqa: E402
from rewardsmith.sim.scene import SceneObject, SceneSpec  # noqa: E402
from mujoco_playground._src import mjx_env  # noqa: E402


WALK = """
import jax.numpy as jnp
def reward(state, action, prev_action):
    return {"forward": jnp.exp(-jnp.square(state["base_lin_vel"][0] - 0.5) / 0.25), "upright": state["up_z"]}
"""


@pytest.fixture(scope="module")
def env():
    if not (mjx_env.MENAGERIE_PATH / "unitree_g1" / "assets").exists():
        pytest.skip("menagerie checkout has no unitree_g1")
    return get_body("g1").make_env(WALK, impl="jax")


def test_body_is_registered():
    body = get_body("g1")
    assert body.capabilities.bipedal and body.capabilities.kind == "humanoid"
    assert (body.nq, body.nv) == (g1.ROBOT_NQ, g1.ROBOT_NV)
    assert set(body.metric_names()) == set(POSE_METRICS)


def test_step_uses_generated_terms_and_no_command(env):
    assert env.term_names == ("forward", "upright")
    state = jax.jit(env.reset)(jax.random.PRNGKey(0))
    assert jnp.all(state.info["command"] == 0)
    step = jax.jit(env.step)
    for _ in range(3):
        state = step(state, jnp.zeros(env.action_size))
    assert jnp.isfinite(state.reward) and float(state.reward) >= 0.0
    assert "reward/tracking_lin_vel" not in state.metrics


def test_features_match_the_spec(env):
    state = jax.jit(env.reset)(jax.random.PRNGKey(1))
    feats = env.features(state.data, state.info, jnp.ones(2, dtype=bool))
    assert set(feats) == {f.name for f in FEATURES}
    for f in FEATURES:
        assert jnp.shape(feats[f.name]) == f.shape, f.name
    assert 0.6 < float(feats["base_height"]) < 0.9


def test_pose_metrics_for_two_feet(env):
    standing = np.tile(np.asarray(env.mj_model.keyframe("knees_bent").qpos), (10, 1))
    out = pose.metrics(env, standing)
    assert set(out) == set(POSE_METRICS)
    assert out["double_support"] == 1.0 and out["flight"] == 0.0 and out["step_count"] == 0.0


def test_describe_lists_every_feature():
    assert all(f'state["{f.name}"]' in describe() for f in FEATURES)


def test_scene_objects_are_refused():
    scene = SceneSpec(objects=(SceneObject("ball", "ball", "red", (0.1,), (1.0, 2.0), (-10.0, 10.0), movable=True),))
    with pytest.raises(RewardError, match="no scene objects"):
        g1.make_env(WALK, impl="jax", scene=scene)
