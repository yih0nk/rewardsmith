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
    result = scene_eval._summarize(np.array([3.0, 3.0]), dists, done, reach_radius=0.5, dt=0.02)
    assert result["target_reached"] == 0.5
    assert result["target_fell"] == 0.5
    assert result["target_min_dist"] == pytest.approx(1.05)
    assert result["target_final_dist"] == 3.0
    assert result["target_time_to_reach"] == 0.04


def test_falling_without_reaching_reports_no_reach_time():
    import numpy as np

    result = scene_eval._summarize(np.array([3.0]), np.array([[2.0], [0.0]]),
                                   np.array([[1], [0]]), reach_radius=0.5, dt=0.02)
    assert result["target_reached"] == 0.0
    assert result["target_time_to_reach"] == -1.0
    assert result["target_final_dist"] == 2.0


@pytest.mark.parametrize("impl", ["jax", "warp"])
def test_batched_rollout_collects_outcome_evidence(tmp_path, impl):
    if impl == "warp":
        import jax
        if jax.default_backend() != "gpu":
            pytest.skip("Warp rollout requires a CUDA GPU")
    if impl != go1.default_impl():
        # Keep JAX and Warp CUDA contexts separate, as the training workers do.
        # Switching backends in one process can invalidate Warp's cached stream.
        import os
        import subprocess
        import sys
        code = "import runpy,sys; from pathlib import Path; runpy.run_path(sys.argv[1])['_check_collected_evidence'](Path(sys.argv[2]),sys.argv[3])"
        result = subprocess.run(
            [sys.executable, "-c", code, str(Path(__file__).resolve()), str(tmp_path), impl],
            env={**os.environ, "XLA_PYTHON_CLIENT_PREALLOCATE": "false"},
            capture_output=True, text=True, timeout=300)
        assert result.returncode == 0, result.stdout + result.stderr
        return
    _check_collected_evidence(tmp_path, impl)


def _check_collected_evidence(tmp_path, impl):
    spec = scene.parse({"objects": [{"name": "step", "kind": "box", "size": [0.6, 0.6, 0.1], "distance": 2}]})
    env = go1.make_env((ROOT / "rewards" / "walk_backwards.py").read_text(), scene=spec, impl=impl)
    metrics = scene_eval.target_metrics(env, standing_policy, None, num_envs=2, steps=5,
                                       success_spec=[{"metric": "climb:step", "op": ">=", "value": 0.8}],
                                       out_dir=tmp_path)
    assert metrics["climb:step"] == 0.0
    assert metrics["scene_fell"] == 0.0
    evidence = json.loads((tmp_path / "scene_evaluation.json").read_text())
    assert len(evidence["episodes"]) == 2
    assert evidence["definitions"]["climb:step"] == ["event", ["climb:step"]]


def test_movable_object_rollout_does_not_credit_an_idle_policy(tmp_path):
    spec = scene.parse({"objects": [
        {"name": "crate", "kind": "box", "size": [0.2, 0.2, 0.2], "distance": 2,
         "bearing": 0, "movable": True},
        {"name": "goal", "kind": "zone", "size": [0.5], "distance": 3, "bearing": 0},
    ]})
    metrics = ["push:crate:goal", "all:avoid:crate&reach:goal", "sequence:reach:crate>reach:goal"]
    env = go1.make_env((ROOT / "rewards" / "walk_backwards.py").read_text(), scene=spec)
    measured = scene_eval.target_metrics(
        env, standing_policy, None, num_envs=2, steps=10,
        success_spec=[{"metric": metric, "op": ">=", "value": 0.8} for metric in metrics],
        out_dir=tmp_path)
    assert all(measured[metric] == 0 for metric in metrics)
    evidence = json.loads((tmp_path / "scene_evaluation.json").read_text())
    assert all(episode["valid"] for episode in evidence["episodes"])
    assert all(not episode["outcomes"][metric]["pass"]
               for episode in evidence["episodes"] for metric in metrics)
