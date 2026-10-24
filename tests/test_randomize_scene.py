import json
from pathlib import Path

import pytest

pytest.importorskip("mujoco_playground")

import jax  # noqa: E402
import numpy as np  # noqa: E402
from mujoco_playground import registry  # noqa: E402

from rewardsmith.sim import go1, scene, train  # noqa: E402

ROOT = Path(__file__).parent.parent
WALK_BACKWARDS = (ROOT / "examples" / "rewards" / "walk_backwards.py").read_text()
FIELDS = ("geom_friction", "body_ipos", "body_mass", "qpos0", "dof_frictionloss", "dof_armature")


def _movable_ball_scene():
    spec = json.loads((ROOT / "examples" / "scenes" / "red_ball.json").read_text())
    spec["objects"][0]["movable"] = True  # a free joint after the robot: 6 more dofs, 7 more qpos
    return scene.parse(spec)


def test_randomizer_matches_playground_on_the_plain_scene():
    model = go1.make_env(WALK_BACKWARDS).mjx_model
    keys = jax.random.split(jax.random.PRNGKey(0), 3)
    ours, our_axes = train.randomizer()(model, keys)
    theirs, their_axes = registry.get_domain_randomizer(go1.BASE_ENV)(model, keys)
    for name in FIELDS:
        np.testing.assert_allclose(getattr(ours, name), getattr(theirs, name), rtol=1e-6, err_msg=name)
        assert getattr(our_axes, name) == getattr(their_axes, name) == 0


def test_randomizer_handles_a_movable_scene_object():
    model = go1.make_env(WALK_BACKWARDS, scene=_movable_ball_scene()).mjx_model
    assert model.nv == go1.ROBOT_NV + 6 and model.nq == go1.ROBOT_NQ + 7

    batched, _ = train.randomizer()(model, jax.random.split(jax.random.PRNGKey(1), 4))

    assert batched.dof_frictionloss.shape == (4, model.nv)
    # The object's free joint is untouched; the robot's joints are randomized.
    np.testing.assert_array_equal(batched.dof_frictionloss[:, go1.ROBOT_NV:], np.broadcast_to(
        model.dof_frictionloss[go1.ROBOT_NV:], (4, 6)))
    np.testing.assert_array_equal(batched.qpos0[:, go1.ROBOT_NQ:], np.broadcast_to(
        model.qpos0[go1.ROBOT_NQ:], (4, 7)))
    assert not np.allclose(batched.qpos0[:, 7:go1.ROBOT_NQ], model.qpos0[7:go1.ROBOT_NQ])
