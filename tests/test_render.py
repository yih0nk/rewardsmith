import io
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mujoco_playground")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from PIL import Image  # noqa: E402

from rewardsmith.sim import go1, render  # noqa: E402
from rewardsmith.sim.train import TrainResult  # noqa: E402

WALK_BACKWARDS = (Path(__file__).parent.parent / "examples" / "rewards" / "walk_backwards.py").read_text()


def test_smoothing_has_no_lag_and_removes_bob():
    t = np.arange(200, dtype=np.float64)
    path = np.stack([0.02 * t, np.zeros_like(t), 0.28 + 0.01 * np.sin(t * 2.0)], axis=1)
    smooth = render._smooth(path, sigma=10)
    inner = slice(40, 160)
    assert np.allclose(smooth[inner, 0], path[inner, 0])  # a steady walk stays centred
    assert np.ptp(smooth[inner, 2]) < 0.1 * np.ptp(path[inner, 2])  # gait bob is gone
    assert smooth.shape == path.shape


def test_side_video_sits_next_to_the_main_one():
    assert render.side_video_path(Path("a/r1c0/rollout.mp4")) == Path("a/r1c0/rollout_side.mp4")


def test_render_model_is_a_copy_sized_for_the_video():
    env = go1.make_env(WALK_BACKWARDS)
    model = render.render_model(env.mj_model, 4320, 7680)
    assert (model.vis.global_.offwidth, model.vis.global_.offheight) == (7680, 4320)
    assert env.mj_model.vis.global_.offwidth < 7680  # the env's own model is untouched
    assert not np.allclose(model.vis.headlight.diffuse, env.mj_model.vis.headlight.diffuse)


def _probe(path: Path) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries",
         "stream=codec_name,pix_fmt,width,height,avg_frame_rate,nb_read_frames", "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    )
    return json.loads(out.stdout)["streams"][0]


def test_rollout_writes_browser_ready_videos_and_small_critic_frames(tmp_path):
    env = go1.make_env(WALK_BACKWARDS)
    stand = TrainResult(
        make_inference_fn=lambda params, deterministic: lambda obs, key: (jnp.zeros(env.action_size), {}),
        params=None,
    )
    video, pngs, summary = render.rollout_and_render(
        env, stand, tmp_path, steps=20, n_frames=4, height=72, width=128,
        critic_height=36, critic_width=64, side_view=True,
    )
    assert video == tmp_path / render.VIDEO_FILE
    assert summary["rollout_steps"] == 20
    for path in (video, render.side_video_path(video)):
        info = _probe(path)
        assert (info["codec_name"], info["pix_fmt"]) == ("h264", "yuv420p")
        assert (info["width"], info["height"]) == (128, 72)
        assert info["avg_frame_rate"] == "50/1"  # one frame per control step
        assert int(info["nb_read_frames"]) == 21  # the reset state plus 20 steps
    assert len(pngs) == 4
    assert all(Image.open(io.BytesIO(p)).size == (64, 36) for p in pngs)


def _stand(env) -> TrainResult:
    return TrainResult(
        make_inference_fn=lambda params, deterministic: lambda obs, key: (0.3 * obs["state"][:12], {}),
        params=None,
    )


def test_scanned_rollout_matches_a_step_by_step_loop():
    env = go1.make_env(WALK_BACKWARDS)
    qpos, summary = render.rollout(env, _stand(env), steps=30)

    policy = _stand(env).make_inference_fn(None, deterministic=True)
    reset, step = jax.jit(env.reset), jax.jit(env.step)
    rng = jax.random.PRNGKey(0)
    state = reset(rng)
    ref = [state.data.qpos]
    for _ in range(30):
        rng, key = jax.random.split(rng)
        state = step(state, policy(state.obs, key)[0])
        ref.append(state.data.qpos)
    assert qpos.shape == (31, env.mj_model.nq)
    np.testing.assert_allclose(qpos, np.stack(ref), atol=1e-4)
    assert summary["rollout_steps"] == 30 and summary["fell"] == 0.0
    assert summary["base_height"] == pytest.approx(float(np.mean(np.stack(ref)[1:, 2])), abs=1e-4)


def test_unknown_camera_is_refused(tmp_path):
    env = go1.make_env(WALK_BACKWARDS)
    with pytest.raises(ValueError, match="unknown camera"):
        render.render_videos(env.mj_model, np.zeros((2, env.mj_model.nq)), env.dt, [(tmp_path / "x.mp4", "nope")],
                             np.array([0]), 72, 128)
