"""Roll out a trained policy, write the mp4, and pick frames for the critic.

Starting point for D: works on CPU and GPU, but camera choice, resolution and
frame picking are first guesses. Keep the function signature stable.
"""

import io
from pathlib import Path

import jax
import jax.numpy as jnp
import mediapy
import numpy as np
from PIL import Image

from .go1 import Go1Env
from .train import TrainResult


def rollout_and_render(
    env: Go1Env,
    result: TrainResult,
    out_dir: Path,
    steps: int = 300,
    n_frames: int = 10,
    height: int = 240,
    width: int = 320,
    camera: str = "track",
    seed: int = 0,
) -> tuple[Path, list[bytes], dict[str, float]]:
    """Return (mp4 path, PNG frames in time order, rollout summary)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    policy = jax.jit(result.make_inference_fn(result.params, deterministic=True))
    reset, step = jax.jit(env.reset), jax.jit(env.step)

    rng = jax.random.PRNGKey(seed)
    state = reset(rng)
    traj, lin, yaw, height_z, up = [state], [], [], [], []
    for _ in range(steps):
        rng, key = jax.random.split(rng)
        action, _ = policy(state.obs, key)
        state = step(state, action)
        traj.append(state)
        lin.append(env.get_local_linvel(state.data))
        yaw.append(env.get_gyro(state.data)[2])
        height_z.append(state.data.qpos[2])
        up.append(env.get_upvector(state.data)[2])
        if bool(state.done):
            break

    video = out_dir / "rollout.mp4"
    frames = env.render(traj[::2], height=height, width=width, camera=camera)
    mediapy.write_video(str(video), frames, fps=1.0 / env.dt / 2)

    picks = np.linspace(0, len(frames) - 1, num=min(n_frames, len(frames))).round().astype(int)
    pngs = [_png(frames[i]) for i in picks]

    lin_a = jnp.stack(lin)
    summary = {
        "forward_velocity": float(lin_a[:, 0].mean()),
        "lateral_velocity": float(lin_a[:, 1].mean()),
        "yaw_rate": float(jnp.stack(yaw).mean()),
        "base_height": float(jnp.stack(height_z).mean()),
        "upright": float(jnp.stack(up).mean()),
        "fell": float(bool(state.done)),
        "rollout_steps": float(len(lin)),
    }
    return video, pngs, summary


def _png(frame: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(frame).save(buf, format="PNG")
    return buf.getvalue()
