from pathlib import Path

import pytest

pytest.importorskip("mujoco_playground")

from rewardsmith.sim import gate  # noqa: E402

WALK_BACKWARDS = (Path(__file__).parent.parent / "examples" / "rewards" / "walk_backwards.py").read_text()
SMALL = {"num_envs": 2, "steps": 8}


def test_valid_reward_passes_with_timing():
    report = gate.run_gate(WALK_BACKWARDS, **SMALL)
    assert report.error is None
    assert report.compile_s > 0 and report.run_s > 0
    assert gate.check_reward(WALK_BACKWARDS, **SMALL) is None


def test_python_if_on_traced_value_gets_jnp_where_hint():
    source = (
        "import jax.numpy as jnp\n"
        "def reward(state, action, prev_action):\n"
        "    if state['up_z'] > 0.5:\n"
        "        up = 1.0\n"
        "    else:\n"
        "        up = 0.0\n"
        "    return {'upright': jnp.float32(up)}\n"
    )
    error = gate.check_reward(source, **SMALL)
    assert error is not None
    assert "jnp.where" in error
    assert "line 3" in error  # points at the offending line in reward.py


def test_nan_reward_names_the_term():
    source = (
        "import jax.numpy as jnp\n"
        "def reward(state, action, prev_action):\n"
        "    return {'height': jnp.log(state['base_height'] - 10.0), 'upright': state['up_z']}\n"
    )
    error = gate.check_reward(source, **SMALL)
    assert error is not None
    assert "'height'" in error and "NaN" in error and "step 0" in error
    assert "'upright'" not in error


def test_syntax_error():
    error = gate.check_reward("def reward(state, action, prev_action:\n    return {}\n", **SMALL)
    assert error is not None and "SyntaxError" in error


def test_shape_bug_is_readable():
    source = (
        "import jax.numpy as jnp\n"
        "def reward(state, action, prev_action):\n"
        "    return {'bad': jnp.sum(state['base_lin_vel'] + state['joint_pos'])}\n"
    )
    error = gate.check_reward(source, **SMALL)
    assert error is not None
    assert "(3,)" in error and "(12,)" in error
    assert "line 3" in error
    assert len(error) <= gate.MAX_MESSAGE


def test_describe_exception_keeps_messages_short():
    exc = ValueError("x" * 5000)
    assert len(gate._clip(gate.describe_exception(exc, ""))) <= gate.MAX_MESSAGE


def test_scene_reward_passes_with_its_scene():
    from rewardsmith.sim import scene

    root = Path(__file__).parent.parent / "examples"
    spec = scene.parse((root / "scenes" / "red_ball.json").read_text())
    source = (root / "rewards" / "walk_to_ball.py").read_text()
    assert gate.check_reward(source, scene=spec, **SMALL) is None
    assert "target_dist" in gate.check_reward(source, **SMALL)  # without the scene there is no target
