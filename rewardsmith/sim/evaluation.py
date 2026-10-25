"""Every rollout after training in one compiled batch: the video, the target metrics, robustness.

Each is ROLLOUT_STEPS control steps of the trained policy. With warp a step costs
about the same for one env as for fifty (the kernels are launch-bound), so three
separate rollouts took ~37 s per candidate on the 4090; one batch of all of them
costs about as much as the largest alone. The batch is

    env 0                the board's video (render.py): the nominal robot from seed 0
    the next TARGET_ENVS the scene episodes, when there is a target or an outcome plan
    the rest             robustness.py's rollouts, when robustness is on

Each env runs the rollout it ran on its own before: the same robot, reset key and
per-step action keys. The one change is in the target episodes, which used to share
one action key per step across the batch and now each draw their own, so their
numbers move by sampling noise only.
"""

import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from . import outcomes, pose, render, robustness, scene_eval
from .base import ROLLOUT_STEPS
from .go1 import Go1Env
from .train import TrainResult

VIDEO_SEED = 0  # render.rollout_and_render's default
TARGET_ENVS = 32  # scene_eval.target_metrics's defaults
TARGET_SEED = 1


def evaluate(
    env: Go1Env,
    result: TrainResult,
    out_dir: Path,
    *,
    robust: bool = True,
    steps: int = ROLLOUT_STEPS,
    success_spec: list[dict] | None = None,
    body: str | None = None,
) -> tuple[Path, list[bytes], dict[str, float], dict[str, float]]:
    """Return (mp4 path, critic PNGs, rollout + target metrics + eval_s, robust_* metrics or {}).
    body names the body env was built from (its randomizer and camera)."""
    own = getattr(env, "evaluate_rollouts", None)
    if own is not None:  # a body with its own task metrics (the arm, sim/panda.py)
        return own(result, out_dir, steps=steps, body=body)
    start = time.perf_counter()
    target_obj = env.scene.target_object
    selected = outcomes.requested(success_spec, env.scene)
    needs_outcomes = bool(selected) or any(c["metric"] == "scene_fell" for c in success_spec or [])
    n_target = TARGET_ENVS if target_obj is not None or needs_outcomes else 0
    plan = robustness.plan() if robust else robustness.plan(n_robots=0, n_seeds=0, pushes=False)
    n_robots = plan.groups.count("robots")

    # robustness.models_for puts the randomized robots first, so the batch is
    # [robots] [video] [target episodes] [seeds, pushes], and everything after the robots
    # is the nominal robot. Plan position i < n_robots is batch row i; later ones shift by 1 + n_target.
    video = n_robots
    target = slice(n_robots + 1, n_robots + 1 + n_target)
    robust_rows = list(range(n_robots)) + list(range(target.stop, target.stop + len(plan.groups) - n_robots))
    target_keys = jax.random.split(jax.random.split(jax.random.PRNGKey(TARGET_SEED))[0], n_target)
    keys = jnp.concatenate([plan.keys[:n_robots], jax.random.PRNGKey(VIDEO_SEED)[None], target_keys, plan.keys[n_robots:]])
    push_dirs = plan.push_dirs[:n_robots] + [(0.0, 0.0)] * (1 + n_target) + plan.push_dirs[n_robots:]
    if n_robots:
        models, in_axes = robustness.models_for(env, n_robots, len(keys) - n_robots, plan.rng_models, body)
    else:
        models, in_axes = env.mjx_model, None
    extra = {"observe": scene_eval.observer(env)} if needs_outcomes else {}
    reset, outs = robustness.run_batch(env, result, models, in_axes, keys, push_dirs, steps, **extra)

    summary, n = render.rollout_summary(*(outs[k][video] for k in ("qpos", "lin", "yaw", "up", "done")))
    summary.update(pose.metrics(env, outs["qpos"][video][:n]))  # feet off the floor, pitch, roll
    robust_metrics: dict[str, float] = {}
    if robust:  # robust_s: the shared batch, which is what robustness costs on its own
        # With the pose metrics, so a pose task's check can pass on the robustness rollouts too.
        rows = [{**robustness.summarize(outs, i), **pose.metrics(env, outs["qpos"][i][:pose.counted(outs["done"][i])])}
                for i in robust_rows]
        # Every row past the robots is the nominal robot, so robustness.json's physics
        # for plan position i can be read from batch row i.
        robust_metrics = robustness.report(plan, rows, models, summary, out_dir=out_dir, start=start)

    qpos = np.concatenate([reset["qpos"][video][None], outs["qpos"][video][:n]])
    video_path, frames = render.render_rollout(
        env, qpos, out_dir, mocap_pos=reset["mocap_pos"][video], mocap_quat=reset["mocap_quat"][video], body=body,
    )
    metrics = dict(summary)
    if target_obj is not None:
        metrics.update(scene_eval.summarize(
            reset["target_dist"][target], outs["target_dist"][target].T, outs["done"][target].T,
            target_obj.reach_radius, env.dt,
        ))
        metrics["scene_fell"] = metrics["target_fell"]
        task_start, task_trace = scene_eval.batch_task_samples(env, reset, outs, target)
        metrics.update(scene_eval.summarize_task(env, task_start, task_trace, outs["done"][target].T))
    if needs_outcomes:
        initial = jax.tree_util.tree_map(lambda value: value[target], reset["scene_outcome"])
        samples = jax.tree_util.tree_map(lambda value: np.swapaxes(value[target], 0, 1), outs["scene_outcome"])
        metrics.update(scene_eval.summarize_outcomes(
            env, initial, samples, selected, seed=TARGET_SEED, out_dir=out_dir))
    metrics["eval_s"] = round(time.perf_counter() - start, 1)  # the batch plus rendering the video
    return video_path, frames, metrics, robust_metrics
