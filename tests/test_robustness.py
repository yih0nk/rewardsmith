import json
from pathlib import Path

import pytest

pytest.importorskip("mujoco_playground")

import jax.numpy as jnp  # noqa: E402

from rewardsmith.sim import go1, robustness  # noqa: E402
from rewardsmith.sim.train import TrainResult  # noqa: E402

WALK_BACKWARDS = (Path(__file__).parent.parent / "examples" / "rewards" / "walk_backwards.py").read_text()
BACKWARDS = {"forward_velocity": -0.6, "lateral_velocity": 0.02, "yaw_rate": 0.05, "fell": 0.0}


def test_main_motion_is_the_clearest_movement():
    assert robustness.main_motion(BACKWARDS) == "forward_velocity"
    assert robustness.main_motion({"forward_velocity": 0.05, "lateral_velocity": 0.0, "yaw_rate": 1.5}) == "yaw_rate"
    assert robustness.main_motion({"forward_velocity": 0.02, "lateral_velocity": 0.03, "yaw_rate": 0.1}) is None


@pytest.mark.parametrize(
    ("rollout", "passes"),
    [
        ({"forward_velocity": -0.5, "fell": 0.0}, True),
        ({"forward_velocity": -0.3, "fell": 0.0}, True),  # exactly half the nominal speed
        ({"forward_velocity": -0.2, "fell": 0.0}, False),  # too slow
        ({"forward_velocity": 0.6, "fell": 0.0}, False),  # wrong way
        ({"forward_velocity": -0.6, "fell": 1.0}, False),  # fell
    ],
)
def test_behavior_proxy(rollout, passes):
    assert robustness.behavior_proxy({**BACKWARDS, **rollout}, BACKWARDS) is passes


def test_standing_behaviors_only_need_to_stay_up():
    still = {"forward_velocity": 0.01, "lateral_velocity": 0.0, "yaw_rate": 0.0, "fell": 0.0}
    assert robustness.behavior_proxy({**still, "forward_velocity": -0.08}, still)
    assert not robustness.behavior_proxy({**still, "fell": 1.0}, still)


@pytest.fixture(scope="module")
def env():
    return go1.make_env(WALK_BACKWARDS, impl="jax")


def _zero_policy(env):
    def make_inference_fn(params, deterministic=True):
        return lambda obs, key: (jnp.zeros(env.action_size), {})

    return TrainResult(make_inference_fn=make_inference_fn, params=None)


def test_evaluate_scores_every_group_and_writes_detail(env, tmp_path):
    standing = {"forward_velocity": 0.0, "lateral_velocity": 0.0, "yaw_rate": 0.0, "fell": 0.0}
    summary = robustness.evaluate(env, _zero_policy(env), standing, tmp_path, n_robots=3, n_seeds=2, steps=120)
    assert summary["robust_robots_n"] == 3 and summary["robust_seeds_n"] == 2 and summary["robust_pushes_n"] == 4
    assert 0 <= summary["robust_robots_pass"] <= 3
    detail = json.loads((tmp_path / "robustness.json").read_text())
    assert [r["group"] for r in detail["rollouts"]] == ["robots"] * 3 + ["seeds"] * 2 + ["pushes"] * 4
    assert [r["name"] for r in detail["rollouts"]][-4:] == ["front", "back", "left", "right"]
    robots, nominal = detail["rollouts"][:3], detail["rollouts"][3:]
    assert len({r["mass_kg"] for r in robots}) == 3  # each randomized robot is different
    assert len({r["mass_kg"] for r in nominal}) == 1  # the rest are the nominal robot


def test_a_custom_check_replaces_the_proxy(env):
    summary = robustness.evaluate(
        env, _zero_policy(env), {"fell": 0.0}, check=lambda m, _: False, n_robots=2, n_seeds=1, pushes=False, steps=5
    )
    assert summary["robust_robots_pass"] == 0 and summary["robust_seeds_pass"] == 0
    assert "robust_pushes_n" not in summary


def test_simulator_scores_robustness_unless_turned_off(monkeypatch):
    from rewardsmith.sim.playground import PlaygroundSimulator

    monkeypatch.delenv("REWARDSMITH_ROBUSTNESS", raising=False)
    assert PlaygroundSimulator(preset="smoke").robustness is True
    monkeypatch.setenv("REWARDSMITH_ROBUSTNESS", "0")
    assert PlaygroundSimulator(preset="smoke").robustness is False
