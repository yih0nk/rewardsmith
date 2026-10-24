import json
from pathlib import Path

import pytest

pytest.importorskip("mujoco_playground")

import jax.numpy as jnp  # noqa: E402

from rewardsmith.sim import go1, scene, scene_eval  # noqa: E402

ROOT = Path(__file__).parent.parent / "examples"


def standing_policy(params, deterministic=True):
    assert deterministic is False
    return lambda obs, key: (jnp.zeros(obs["state"].shape[:-1] + (12,)), {})


def test_standing_policy_never_reaches_the_ball():
    spec = scene.parse(json.loads((ROOT / "scenes" / "red_ball.json").read_text()))
    env = go1.make_env((ROOT / "rewards" / "walk_to_ball.py").read_text(), scene=spec)
    m = scene_eval.target_metrics(env, standing_policy, None, num_envs=4, steps=20)
    assert m["target_reached"] == 0.0 and m["target_time_to_reach"] == -1.0
    assert 1.4 < m["target_start_dist"] < 3.1
    assert m["target_min_dist"] == pytest.approx(m["target_start_dist"], abs=0.3)
    assert m["target_fell"] == 0.0


def test_no_target_no_metrics():
    env = go1.make_env((ROOT / "rewards" / "walk_backwards.py").read_text())
    assert scene_eval.target_metrics(env, standing_policy, None) == {}



def test_target_metrics_ignore_sliding_after_the_first_fall():
    import numpy as np

    # First robot falls at step 2, then slides into reach; only the second robot succeeds.
    dists = np.array([[3.0, 3.0], [2.0, 0.1], [0.1, 3.0], [0.0, 4.0]])
    done = np.array([[0, 0], [1, 0], [0, 0], [0, 0]])
    result = scene_eval.summarize(np.array([3.0, 3.0]), dists, done, reach_radius=0.5, dt=0.02)
    assert result["target_reached"] == 0.5
    assert result["target_fell"] == 0.5
    assert result["target_min_dist"] == pytest.approx(1.05)
    assert result["target_final_dist"] == 3.0
    assert result["target_time_to_reach"] == 0.04


def test_falling_without_reaching_reports_no_reach_time():
    import numpy as np

    result = scene_eval.summarize(np.array([3.0]), np.array([[2.0], [0.0]]),
                                   np.array([[1], [0]]), reach_radius=0.5, dt=0.02)
    assert result["target_reached"] == 0.0
    assert result["target_time_to_reach"] == -1.0
    assert result["target_final_dist"] == 2.0
