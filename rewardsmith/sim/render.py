"""Roll out a trained policy, write the board's mp4s, and pick frames for the critic.

The board video is 1280x720 at 50 fps (one frame per control step), H.264 in
yuv420p so every browser plays it. A chase camera follows the base with
zero-lag smoothing at a fixed 3/4 angle, so turns and spins read as turns and
spins instead of the world rotating. An optional side view goes to
rollout_side.mp4 next to the main video. The critic gets small PNGs of the
same rollout, which keeps Claude calls cheap.
"""

import contextlib
import copy
import io
from dataclasses import dataclass
from pathlib import Path

import jax
import mediapy
import mujoco
import numpy as np
from PIL import Image

from .go1 import Go1Env
from .train import TrainResult

VIDEO_FILE = "rollout.mp4"
SIDE_VIDEO_FILE = "rollout_side.mp4"

# x264 settings: visually lossless for this simple scene, fast enough to keep up with rendering.
# faststart puts the index first so the board can start playing before the file has loaded.
ENCODE = dict(codec="h264", crf=20, encoded_format="yuv420p")
ENCODE_ARGS = ["-preset", "veryfast", "-profile:v", "high", "-g", "50", "-movflags", "+faststart"]


@dataclass(frozen=True)
class FollowCam:
    """A free camera aimed at the smoothed base position from a fixed world angle."""

    azimuth: float  # degrees; 0 looks along +x, 90 along +y (the robot starts facing +x)
    elevation: float  # degrees; negative looks down
    distance: float  # metres from the look-at point
    smooth_s: float = 0.35  # std of the Gaussian that smooths the base path, in seconds


CAMERAS = {
    # In front of the robot and to its right, looking back at it, a little from above.
    "chase": FollowCam(azimuth=140.0, elevation=-18.0, distance=1.35),
    # Square to the robot's starting left side, near ground level: gait, height and flips.
    "side": FollowCam(azimuth=-90.0, elevation=-14.0, distance=1.5),
}


def rollout_and_render(
    env: Go1Env,
    result: TrainResult,
    out_dir: Path,
    steps: int = 500,
    n_frames: int = 10,
    height: int = 720,
    width: int = 1280,
    camera: str = "chase",
    seed: int = 0,
    *,
    critic_height: int = 288,
    critic_width: int = 512,
    side_view: bool = False,
) -> tuple[Path, list[bytes], dict[str, float]]:
    """Return (mp4 path, PNG frames in time order, rollout summary).

    `camera` is a key of CAMERAS or a camera name from the Go1 model ("track",
    "side", "back", "top"). With side_view, also writes SIDE_VIDEO_FILE; find it
    with side_video_path(mp4 path).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    qpos, summary, mocap_pos, mocap_quat = _rollout(env, result, steps, seed)

    video = out_dir / VIDEO_FILE
    views = [(video, camera)] + ([(side_video_path(video), "side")] if side_view else [])
    picks = np.linspace(0, len(qpos) - 1, num=min(n_frames, len(qpos))).round().astype(int)
    stills = render_videos(
        env.mj_model, qpos, env.dt, views, picks, height, width,
        mocap_pos=mocap_pos, mocap_quat=mocap_quat,
    )
    pngs = [_png(f, critic_width, critic_height) for f in stills]
    return video, pngs, summary


def rollout(env: Go1Env, result: TrainResult, steps: int, seed: int = 0) -> tuple[np.ndarray, dict[str, float]]:
    """Run the policy for up to `steps` control steps, stopping at the first fall.

    The policy samples its actions (seeded, so reruns match), as in brax's PPO eval.
    The mean action alone is not what was trained: spin policies fall over with it.

    Returns qpos for the reset state and every step taken, and the rollout summary.
    The whole rollout is one compiled scan: far cheaper than a Python loop that
    dispatches and syncs every step, especially with four candidates sharing the GPU.
    """
    qpos, summary, _, _ = _rollout(env, result, steps, seed)
    return qpos, summary


def _rollout(env: Go1Env, result: TrainResult, steps: int, seed: int = 0):
    """Keep reset mocap transforms too: fixed objects live outside qpos."""
    policy = result.make_inference_fn(result.params, deterministic=False)

    def body(carry, _):
        state, rng = carry
        rng, key = jax.random.split(rng)
        action, _ = policy(state.obs, key)
        state = env.step(state, action)
        d = state.data
        out = (d.qpos, env.get_local_linvel(d), env.get_gyro(d)[2], env.get_upvector(d)[2], state.done)
        return (state, rng), out

    @jax.jit
    def run(rng):
        state = env.reset(rng)
        _, outs = jax.lax.scan(body, (state, rng), length=steps)
        return state.data.qpos, state.data.mocap_pos, state.data.mocap_quat, outs

    qpos0, mocap_pos, mocap_quat, outs = jax.device_get(run(jax.random.PRNGKey(seed)))
    qpos, lin, yaw, up, done = (np.asarray(x) for x in outs)
    fell = bool(done.any())
    n = int(np.argmax(done)) + 1 if fell else steps  # keep the step that fell, drop the ones after
    summary = {
        "forward_velocity": float(lin[:n, 0].mean()),
        "lateral_velocity": float(lin[:n, 1].mean()),
        "yaw_rate": float(yaw[:n].mean()),
        "base_height": float(qpos[:n, 2].mean()),
        "upright": float(up[:n].mean()),
        "fell": float(fell),
        "rollout_steps": float(n),
    }
    return np.concatenate([qpos0[None], qpos[:n]]), summary, mocap_pos, mocap_quat


def side_video_path(video: Path) -> Path:
    """Where rollout_and_render(side_view=True) puts the side view of `video`."""
    return video.with_name(SIDE_VIDEO_FILE)


def render_videos(
    mj_model: mujoco.MjModel,
    qpos: np.ndarray,
    dt: float,
    views: list[tuple[Path, str]],
    picks: np.ndarray,
    height: int,
    width: int,
    *,
    mocap_pos: np.ndarray | None = None,
    mocap_quat: np.ndarray | None = None,
) -> list[np.ndarray]:
    """Write one mp4 per (path, camera) view, a frame per qpos row. Return the first view's frames at `picks`."""
    model = render_model(mj_model, height, width)
    data = mujoco.MjData(model)
    if mocap_pos is not None:
        data.mocap_pos[:] = mocap_pos
    if mocap_quat is not None:
        data.mocap_quat[:] = mocap_quat
    tracks = [_camera(model, name, qpos, dt) for _, name in views]
    picked = set(picks.tolist())
    stills = []
    with contextlib.ExitStack() as stack:
        renderers = [stack.enter_context(mujoco.Renderer(model, height=height, width=width)) for _ in views]
        writers = [
            stack.enter_context(
                mediapy.VideoWriter(path, shape=(height, width), fps=1.0 / dt, ffmpeg_args=ENCODE_ARGS, **ENCODE)
            )
            for path, _ in views
        ]
        for i, q in enumerate(qpos):
            data.qpos[:] = q
            mujoco.mj_forward(model, data)
            for v, (renderer, writer, track) in enumerate(zip(renderers, writers, tracks)):
                renderer.update_scene(data, camera=track(i))
                frame = renderer.render()
                writer.add_image(frame)
                if v == 0 and i in picked:
                    stills.append(frame.copy())
    return stills


def render_model(mj_model: mujoco.MjModel, height: int, width: int) -> mujoco.MjModel:
    """A copy of the model with render-only quality settings. Physics is untouched."""
    model = copy.deepcopy(mj_model)
    vis = model.vis
    vis.global_.offwidth = max(vis.global_.offwidth, width)
    vis.global_.offheight = max(vis.global_.offheight, height)
    vis.quality.offsamples = 8  # MSAA: clean edges on legs and grid lines
    vis.quality.shadowsize = max(vis.quality.shadowsize, 8192)
    # Softer headlight so the spotlight's shadows read; the scene's default washes them out.
    vis.headlight.ambient[:] = 0.35
    vis.headlight.diffuse[:] = 0.45
    vis.headlight.specular[:] = 0.15
    return model


def _camera(model: mujoco.MjModel, name: str, qpos: np.ndarray, dt: float):
    """Return frame index -> the camera to render it with (a model camera id, or a free camera)."""
    if name not in CAMERAS:
        cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name)
        if cam_id < 0:
            raise ValueError(f"unknown camera {name!r}: use one of {sorted(CAMERAS)} or a model camera")
        return lambda i: cam_id
    spec = CAMERAS[name]
    lookat = _smooth(qpos[:, :3], spec.smooth_s / dt)
    lookat[:, 2] = np.maximum(lookat[:, 2] - 0.12, 0.08)  # aim between trunk and feet
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.azimuth, cam.elevation, cam.distance = spec.azimuth, spec.elevation, spec.distance

    def at(i: int) -> mujoco.MjvCamera:
        cam.lookat[:] = lookat[i]
        return cam

    return at


def _smooth(x: np.ndarray, sigma: float) -> np.ndarray:
    """Zero-lag Gaussian smoothing along axis 0, holding the ends."""
    if sigma <= 0 or len(x) < 2:
        return x.astype(np.float64, copy=True)
    r = int(3 * sigma)
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    padded = np.pad(x.astype(np.float64), ((r, r), (0, 0)), mode="edge")
    return np.stack([np.convolve(padded[:, j], k, mode="valid") for j in range(x.shape[1])], axis=1)


def _png(frame: np.ndarray, width: int, height: int) -> bytes:
    image = Image.fromarray(frame)
    if image.size != (width, height):
        image = image.resize((width, height), Image.LANCZOS)
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()
