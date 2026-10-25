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
    measured = _shared_evaluation(env, tmp_path, ["climb:step"])
    assert measured["climb:step"] == 0 and measured["scene_fell"] == 0


def _shared_evaluation(env, tmp_path, metrics):
    from rewardsmith.sim import evaluation, render, robustness
    from rewardsmith.sim.train import TrainResult

    batches = []
    real = robustness.run_batch
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(evaluation, "TARGET_ENVS", 2)
        patch.setattr(render, "render_rollout", lambda env, qpos, path, **kw: (path / "rollout.mp4", []))
        patch.setattr(robustness, "run_batch", lambda *args, **kw: batches.append(len(args[4])) or real(*args, **kw))
        _, _, measured, _ = evaluation.evaluate(
            env, TrainResult(make_inference_fn=standing_policy, params=None), tmp_path,
            robust=False, steps=5,
            success_spec=[{"metric": metric, "op": ">=", "value": 0.8} for metric in metrics])
    assert batches == [3]  # video plus two scene episodes, in one physics batch
    evidence = json.loads((tmp_path / "scene_evaluation.json").read_text())
    assert len(evidence["episodes"]) == 2
    assert all(episode["valid"] for episode in evidence["episodes"])
    return measured


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
    batched = _shared_evaluation(env, tmp_path, metrics)
    assert all(batched[metric] == 0 for metric in metrics)


# --- task metrics: where the object ended up, and which phase each episode got stuck at ---

def _task(start_dist, dists, done, **kw):
    """summarize_task_arrays with base-to-target distances; the base stands still."""
    import numpy as np

    dists = np.asarray(dists, dtype=float)
    trace = {"dist": dists, "speed": np.zeros_like(dists), "obj_xy": np.zeros(dists.shape + (2,))}
    trace.update(kw.pop("trace", {}))
    start = {"dist": np.asarray(start_dist, dtype=float), "obj_xy": np.zeros((dists.shape[1], 2))}
    start.update(kw.pop("start", {}))
    return scene_eval.summarize_task_arrays(start, trace, np.asarray(done), **{"dt": 0.02, **kw})


def test_reach_phases_ignore_sliding_after_the_first_fall():
    import numpy as np

    dists = np.array([[3.0, 3.0], [2.0, 0.1], [0.1, 3.0], [0.0, 4.0]])
    done = np.array([[0, 0], [1, 0], [0, 0], [0, 0]])
    result = _task([3.0, 3.0], dists, done, reach_radius=0.5)
    assert result["phase_no_approach"] == 0.5 and result["phase_left"] == 0.5 and result["phase_arrived"] == 0.0


def test_arriving_and_stopping_counts_as_arrived():
    import numpy as np

    dists = np.array([[2.0], [1.0], [0.3], [0.3]])
    speed = np.array([[1.0], [1.0], [0.0], [0.0]])
    result = _task([2.5], dists, np.zeros((4, 1)), reach_radius=0.5, trace={"speed": speed})
    assert result["target_final_in"] == 1.0 and result["phase_arrived"] == 1.0
    assert result["target_near_frac"] == 0.5
    assert result["final_speed"] == pytest.approx(0.5)  # 1 s tail is 50 steps: all 4 here


def _goal_episode(goal_dists, obj_moved, base_dists, goal_start):
    """One push episode: object-to-goal distance per step, the object's final displacement
    along x, base-to-object distance per step, and the object-to-goal distance at the start."""
    import numpy as np

    obj = np.zeros((len(goal_dists), 2))
    obj[-1, 0] = obj_moved
    return np.asarray(goal_dists, float), obj, np.asarray(base_dists, float), goal_start


def test_push_task_phases_say_where_each_episode_got_stuck():
    import numpy as np

    far, near = [1.5, 1.5, 1.5], [1.5, 0.4, 0.4]
    episodes = (
        _goal_episode([2.0, 2.0, 2.0], 0.0, far, 2.0),   # never went to the ball
        _goal_episode([2.0, 2.0, 2.0], 0.05, near, 2.0),  # got there, ball barely moved
        _goal_episode([2.0, 2.4, 2.6], 0.6, near, 2.0),   # pushed it away from the goal
        _goal_episode([2.0, 1.5, 1.2], 0.8, near, 2.0),   # pushed toward, stopped short
        _goal_episode([2.0, 0.3, 0.9], 1.5, near, 2.0),   # in the goal, then out again
        _goal_episode([2.0, 1.0, 0.2], 1.8, near, 2.0),   # delivered
    )
    g, obj, base = (np.stack([e[i] for e in episodes], axis=1) for i in range(3))
    result = _task(np.full(6, 1.5), base, np.zeros_like(base), reach_radius=0.5, goal_radius=0.5, movable=True,
                   trace={"goal_dist": g, "obj_xy": obj}, start={"goal_dist": np.array([e[3] for e in episodes])})
    for p in scene_eval.GOAL_PHASES:
        assert result[f"phase_{p}"] == pytest.approx(1 / 6), p
    assert result["object_goal_reached"] == pytest.approx(2 / 6)
    assert result["object_in_goal_final"] == pytest.approx(1 / 6)
    assert result["object_goal_start_dist"] == 2.0
    assert result["object_goal_min_dist"] == pytest.approx((2 + 2 + 2 + 1.2 + 0.3 + 0.2) / 6)
    assert result["object_goal_progress"] == pytest.approx(((0 + 0 - 0.6 + 0.8 + 1.1 + 1.8) / 2) / 6)
    assert result["object_moved"] == pytest.approx((0 + 0.05 + 0.6 + 0.8 + 1.5 + 1.8) / 6)


PUSH_SCENE = {
    "objects": [
        {"name": "ball", "kind": "ball", "color": "orange", "size": [0.15], "distance": [1.0, 1.5],
         "bearing": [-20, 20], "movable": True, "mass": 0.4},
        {"name": "pad", "kind": "zone", "color": "green", "size": [0.5], "distance": [2.5, 3.0],
         "bearing": [-20, 20]},
    ],
    "target": "ball",
}


def test_push_scene_reports_object_metrics_on_both_paths(tmp_path):
    spec = scene.parse(PUSH_SCENE)
    assert spec.goal == "pad"
    env = go1.make_env((ROOT / "rewards" / "push_to_zone.py").read_text(), scene=spec)
    m = scene_eval.target_metrics(env, standing_policy, None, num_envs=4, steps=20)
    assert m["object_moved"] < 0.05
    assert m["object_in_goal_final"] == 0.0 and m["phase_no_approach"] == 1.0
    assert 0.5 < m["object_goal_start_dist"] < 4.5

    from rewardsmith.sim import evaluation, render
    from rewardsmith.sim.train import TrainResult

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(evaluation, "TARGET_ENVS", 4)
        patch.setattr(render, "render_rollout", lambda env, qpos, path, **kw: (path / "rollout.mp4", []))
        _, _, batched, _ = evaluation.evaluate(
            env, TrainResult(make_inference_fn=standing_policy, params=None), tmp_path, robust=False, steps=20)
    assert batched["object_moved"] < 0.05 and batched["phase_no_approach"] == 1.0
    assert batched["object_goal_start_dist"] == pytest.approx(m["object_goal_start_dist"], abs=1.5)
    assert set(m) - {"scene_fell"} <= set(batched) | {"scene_fell"}
