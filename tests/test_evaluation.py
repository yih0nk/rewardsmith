import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mujoco_playground")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from rewardsmith.sim import evaluation, go1, render, robustness, scene  # noqa: E402
from rewardsmith.sim.train import TrainResult  # noqa: E402

ROOT = Path(__file__).parent.parent / "examples"
STEPS = 12


def _noisy(params, deterministic):
    assert deterministic is False  # the trained gait is the sampled one
    return lambda obs, key: (0.3 * obs["state"][..., :12] + 0.2 * jax.random.normal(key, obs["state"].shape[:-1] + (12,)), {})


@pytest.fixture(scope="module")
def ball_env():
    spec = scene.parse(json.loads((ROOT / "scenes" / "red_ball.json").read_text()))
    return go1.make_env((ROOT / "rewards" / "walk_to_ball.py").read_text(), impl="jax", scene=spec)


def _no_render(monkeypatch):
    calls = []

    def render_rollout(env, qpos, out_dir, **kw):
        calls.append((qpos, kw))
        return out_dir / render.VIDEO_FILE, [b"png"]

    monkeypatch.setattr(render, "render_rollout", render_rollout)
    return calls


def test_one_batch_reproduces_the_separate_video_and_robustness_rollouts(ball_env, tmp_path, monkeypatch):
    result = TrainResult(make_inference_fn=_noisy, params=None)
    qpos_ref, summary_ref, mocap_ref, _ = render._rollout(ball_env, result, STEPS, evaluation.VIDEO_SEED)
    (tmp_path / "ref").mkdir()
    robust_ref = robustness.evaluate(ball_env, result, summary_ref, tmp_path / "ref", steps=STEPS)
    rows_ref = json.loads((tmp_path / "ref" / "robustness.json").read_text())["rollouts"]
    calls = _no_render(monkeypatch)

    (tmp_path / "new").mkdir()
    video, frames, metrics, robust = evaluation.evaluate(ball_env, result, tmp_path / "new", steps=STEPS)

    assert video == tmp_path / "new" / render.VIDEO_FILE and frames == [b"png"]
    [(qpos, kw)] = calls
    np.testing.assert_allclose(qpos, qpos_ref, atol=1e-4)  # the same video, frame for frame
    np.testing.assert_allclose(kw["mocap_pos"], mocap_ref, atol=1e-5)
    for key, value in summary_ref.items():
        assert metrics[key] == pytest.approx(value, abs=1e-4)
    assert set(metrics) >= {"target_reached", "target_start_dist", "target_min_dist", "target_fell"}
    assert 1.4 < metrics["target_start_dist"] < 3.1
    assert {k: v for k, v in robust.items() if k != "robust_s"} == {k: v for k, v in robust_ref.items() if k != "robust_s"}
    rows = json.loads((tmp_path / "new" / "robustness.json").read_text())["rollouts"]
    assert [r["name"] for r in rows] == [r["name"] for r in rows_ref]
    for row, ref in zip(rows, rows_ref):
        for key in ("forward_velocity", "yaw_rate", "base_height", "fell", "rollout_steps", "mass_kg", "floor_friction"):
            assert row[key] == pytest.approx(ref[key], abs=1e-4), (row["name"], key)


def test_without_robustness_or_a_target_only_the_video_env_runs(tmp_path, monkeypatch):
    env = go1.make_env((ROOT / "rewards" / "walk_backwards.py").read_text(), impl="jax")
    batches = []
    real = robustness.run_batch
    monkeypatch.setattr(robustness, "run_batch", lambda *a, **k: batches.append(len(a[4])) or real(*a, **k))
    _no_render(monkeypatch)

    _, _, metrics, robust = evaluation.evaluate(
        env, TrainResult(make_inference_fn=_noisy, params=None), tmp_path, robust=False, steps=STEPS
    )

    assert batches == [1] and robust == {}
    assert "target_reached" not in metrics and metrics["rollout_steps"] <= STEPS
    assert not (tmp_path / "robustness.json").exists()
