import json
from pathlib import Path

import pytest

pytest.importorskip("mujoco_playground")

import jax.numpy as jnp  # noqa: E402

from rewardsmith.sim import go1, scene, scene_eval  # noqa: E402

ROOT = Path(__file__).parent.parent / "examples"


def standing_policy(params, deterministic=True):
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
