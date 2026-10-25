import numpy as np
import pytest

from rewardsmith.agents import checker
from rewardsmith.sim import panda_spec
from rewardsmith.sim.bodies.panda import PANDA
from rewardsmith.sim.bodies.base import Body

LIFT = """import jax.numpy as jnp


def reward(state, action, prev_action):
    return {"reach": 1.0 - jnp.tanh(5.0 * state["gripper_to_cube_dist"]),
            "lift": state["grasped"] * state["cube_height"]}
"""


def test_body_is_a_body():
    assert isinstance(PANDA, Body)
    assert PANDA.capabilities.can_grasp and not PANDA.capabilities.can_walk and PANDA.capabilities.legs == 0
    assert "chase" in PANDA.camera and "lookat" in PANDA.camera["chase"]
    assert set(PANDA.metric_names()) == set(panda_spec.ARM_METRICS)


def test_describe_lists_every_feature_and_no_legs():
    text = PANDA.describe()
    for f in panda_spec.FEATURES:
        assert f'state["{f.name}"]' in text
    assert "shape (4,)" in text and "foot" not in text


def test_env_features_match_the_spec():
    jax = pytest.importorskip("jax")
    from rewardsmith.sim import panda

    env = PANDA.make_env(LIFT, impl="jax", mode="steer")
    assert env.action_size == 4 and env.term_names == ("lift", "reach")
    state = env.reset(jax.random.PRNGKey(0))
    feats = env.features(state.data, state.info)
    assert {k: tuple(v.shape) for k, v in feats.items()} == {f.name: f.shape for f in panda_spec.FEATURES}
    state = env.step(state, jax.numpy.zeros(4))
    assert float(state.done) == 0.0 and np.isfinite(float(state.reward))
    assert set(panda.dummy_state()) == set(feats)


def _frames(height, held, dist_to_target, fell=None, n=2, t=50):
    cube = np.zeros((n, t, 3))
    cube[..., 2] = panda_spec.CUBE_REST_Z + height
    target = cube.copy()
    target[..., 0] += dist_to_target
    return {"cube_pos": cube, "target_pos": target, "grasped": np.full((n, t), float(held)),
            "cube_height": np.full((n, t), height), "fell": np.zeros((n, t)) if fell is None else fell}


def test_rollout_metrics():
    from rewardsmith.sim import panda

    lifted = panda.rollout_metrics(_frames(0.2, True, 0.0))
    assert lifted["lifted"] == 1.0 and lifted["at_target"] == 1.0 and lifted["fell"] == 0.0
    assert lifted["lift_held_s"] == pytest.approx(50 * panda_spec.CTRL_DT)
    resting = panda.rollout_metrics(_frames(0.0, False, 0.3))
    assert resting["lifted"] == 0.0 and resting["at_target"] == 0.0 and resting["final_target_dist"] == pytest.approx(0.3)
    fell = np.zeros((2, 50))
    fell[0, 10] = 1.0
    assert panda.rollout_metrics(_frames(0.0, False, 0.3, fell))["fell"] == 0.5


def test_arm_checks():
    spec = checker.validate_arm(
        [{"metric": "lifted", "op": ">=", "value": 0.8}, {"metric": "fell", "op": "<=", "value": 0.1}], "pick it up")
    assert spec[0]["metric"] == "lifted"
    with pytest.raises(checker.SpecError):
        checker.validate_arm([{"metric": "fell", "op": "<=", "value": 0.1}], "pick it up")
    with pytest.raises(checker.SpecError):
        checker.validate_arm([{"metric": "forward_velocity", "op": ">", "value": 0.3}], "pick it up")
    assert checker.evaluate(checker.default_arm_check(None), {"lifted": 0.9, "max_cube_height": 0.2, "fell": 0.0})["pass"]


def test_arm_specs_pass_the_scene_validator():
    spec = [{"metric": "placed", "op": ">=", "value": 0.8}, {"metric": "fell", "op": "<=", "value": 0.1}]
    assert checker.validate_for_scene(spec, "push the block to the line", None) == spec
    assert not checker.is_arm_spec([{"metric": "fell", "op": "<", "value": 0.5}])
