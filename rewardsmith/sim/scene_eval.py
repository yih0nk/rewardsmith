"""Did the trained policy reach the target? Measured over many episodes, not one video.

Runs the seeded, sampled policy in a batch of fresh episodes (objects placed
afresh in each) and reports distances to the target object. The critic reads
these alongside the frames, and they are the headline number for goal tasks.

summarize_task adds how the task went: for "move the target to the goal" tasks,
where the OBJECT ended up relative to the goal, and the fraction of episodes stuck
at each phase of the task, so the critic and editor know which phase to fix.
"""

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from . import outcomes
from .go1 import Go1Env
from .base import ROLLOUT_STEPS
from .go1_spec import CTRL_DT

MOVED = 0.15  # m; an object displaced less than this was not really moved
FINAL_S = 1.0  # final_speed averages the base speed over this last stretch, s

# Per-episode phases of a "move the target to the goal" task, in task order. Each episode
# lands in exactly one, so the fractions say which step the policy gets stuck at.
GOAL_PHASES = ("no_approach", "no_push", "wrong_way", "short", "overshoot", "delivered")
# Per-episode phases of a "go to the target" task (a fixed target, or no goal).
REACH_PHASES = ("no_approach", "left", "arrived")


def target_metrics(
    env: Go1Env,
    make_policy: Callable[..., Any],
    params: Any,
    num_envs: int = 32,
    steps: int = ROLLOUT_STEPS,
    seed: int = 1,
    success_spec: list[dict[str, Any]] | None = None,
    out_dir=None,
) -> dict[str, float]:
    """Metrics about the scene's target object, or {} if the scene has none.

    target_reached    fraction of episodes where the base got within reach (see SceneObject.reach_radius)
    target_start_dist mean distance at the start, m
    target_min_dist   mean closest distance, m
    target_final_dist mean distance at the end, m
    target_time_to_reach  mean seconds until first in reach, over episodes that reached it
    target_fell       fraction of episodes that ended with the robot flipped
    """
    target = env.scene.target_object
    selected = outcomes.requested(success_spec, env.scene)
    needs_outcomes = bool(selected) or any(c["metric"] == "scene_fell" for c in success_spec or [])
    if target is None and not needs_outcomes:
        return {}
    policy = make_policy(params, deterministic=False)
    observe = observer(env)

    def run(rng):
        reset_rng, act_rng = jax.random.split(rng)
        state = jax.vmap(env.reset)(jax.random.split(reset_rng, num_envs))
        start = jax.vmap(observe)(state)

        def body(state, key):
            action, _ = policy(state.obs, key)
            state = jax.vmap(env.step)(state, action)
            return state, jax.vmap(observe)(state)

        _, samples = jax.lax.scan(body, state, jax.random.split(act_rng, steps))
        return start, samples

    start, samples = jax.device_get(jax.jit(run)(jax.random.PRNGKey(seed)))
    summary = summarize(start["distance"], samples["distance"], samples["done"], target.reach_radius, env.dt) if target else {}
    if target:
        summary["scene_fell"] = summary["target_fell"]
        task_start, task_trace = _task_samples(env, start, samples)
        summary.update(summarize_task(env, task_start, task_trace, samples["done"]))
    if needs_outcomes:
        summary.update(summarize_outcomes(env, start, samples, selected, seed=seed, out_dir=out_dir))
    return summary


def observer(env: Go1Env):
    target = env.scene.target_object
    placed = [(obj.name, env.mj_model.body("obj_" + obj.name).id) for obj in env.scene.objects]

    def observe(state):
        data = state.data
        objects = {}
        for name, body in placed:
            forward = data.xmat[body][:, 0]
            objects[name] = {"pos": data.xpos[body], "yaw": jnp.arctan2(forward[1], forward[0]),
                             "rotation": data.xmat[body]}
        return {
            "base": data.qpos[:3], "yaw": env._yaw(data), "up": env.get_upvector(data)[2],
            "speed": jnp.linalg.norm(env.get_local_linvel(data)[:2]),
            "yaw_rate": env.get_gyro(data)[2],
            "feet": data.site_xpos[env._feet_site_id],
            "contacts": state.info["last_contact"], "objects": objects, "done": state.done,
            "distance": env.scene_features(data)["target_dist"] if target else jnp.float32(0),
        }

    return observe


def summarize_outcomes(env, start, samples, selected, *, seed, out_dir=None):
    steps, num_envs = samples["done"].shape
    def row(tree, *indices):
        return jax.tree_util.tree_map(lambda array: np.asarray(array)[indices].tolist(), tree)
    episodes = [{"start": row(start, i), "frames": [row(samples, t, i) for t in range(steps)]}
                for i in range(num_envs)]
    measured, evidence = outcomes.evaluate(env.scene, episodes, env.dt, selected)
    if out_dir is not None:
        import json
        (out_dir / "scene_evaluation.json").write_text(json.dumps({
            "dt": env.dt, "steps": steps, "seed": seed, "episodes": evidence,
            "definitions": {name: outcomes.parts(name, env.scene) for name in selected},
        }, indent=2))
    return measured


def summarize(start, dists, done, reach_radius: float, dt: float) -> dict[str, float]:
    """start (num_envs,), dists and done (steps, num_envs). Only observations up to and including each episode's first terminal step count."""
    done = np.asarray(done, dtype=bool)
    valid = np.concatenate([np.ones_like(done[:1]), np.cumprod(~done[:-1], axis=0).astype(bool)])
    reached = (dists <= reach_radius) & valid
    any_reached = reached.any(axis=0)
    first = reached.argmax(axis=0)
    fell = done.any(axis=0)
    final_index = np.where(fell, done.argmax(axis=0), len(dists) - 1)
    final = np.take_along_axis(dists, final_index[None], axis=0)[0]
    return {
        "target_reached": float(any_reached.mean()),
        "target_start_dist": float(start.mean()),
        "target_min_dist": float(np.where(valid, dists, np.inf).min(axis=0).mean()),
        "target_final_dist": float(final.mean()),
        "target_time_to_reach": float(((first[any_reached] + 1) * dt).mean()) if any_reached.any() else -1.0,
        "target_fell": float(fell.mean()),
    }


def _task_samples(env: Go1Env, start, samples):
    """summarize_task's inputs from the observer's telemetry."""
    target, goal = env.scene.target, env.scene.goal

    def pick(tree):
        out = {"dist": np.asarray(tree["distance"]), "speed": np.asarray(tree["speed"]),
               "obj_xy": np.asarray(tree["objects"][target]["pos"])[..., :2]}
        if goal is not None:
            out["goal_dist"] = np.linalg.norm(out["obj_xy"] - np.asarray(tree["objects"][goal]["pos"])[..., :2], axis=-1)
        return out

    return pick(start), pick(samples)


def batch_task_samples(env: Go1Env, reset: dict, outs: dict, rows: slice):
    """summarize_task's inputs from robustness.run_batch's arrays (evaluation.py), for batch rows.

    The movable target's position is its free joint in qpos; a goal never moves, so its
    position is the mocap position it was placed at."""
    scene = env.scene
    placed = {p.obj.name: p for p in env._placed}
    target, goal = placed[scene.target], placed.get(scene.goal)
    qpos0, qpos = reset["qpos"][rows], np.swapaxes(outs["qpos"][rows], 0, 1)  # (envs, nq), (steps, envs, nq)

    def obj_xy(q):
        if target.qpos_adr >= 0:
            return q[..., target.qpos_adr:target.qpos_adr + 2]
        return np.broadcast_to(reset["mocap_pos"][rows][:, target.mocap_id, :2], q.shape[:-1] + (2,))

    start = {"dist": reset["target_dist"][rows], "obj_xy": obj_xy(qpos0)}
    trace = {"dist": np.swapaxes(outs["target_dist"][rows], 0, 1), "obj_xy": obj_xy(qpos),
             "speed": np.linalg.norm(np.swapaxes(outs["lin"][rows], 0, 1)[..., :2], axis=-1)}
    if goal is not None:
        goal_xy = reset["mocap_pos"][rows][:, goal.mocap_id, :2]
        start["goal_dist"] = np.linalg.norm(start["obj_xy"] - goal_xy, axis=-1)
        trace["goal_dist"] = np.linalg.norm(trace["obj_xy"] - goal_xy[None], axis=-1)
    return start, trace


def summarize_task(env: Go1Env, start: dict, trace: dict, done) -> dict[str, float]:
    """summarize_task_arrays with the scene's radii."""
    scene = env.scene
    return summarize_task_arrays(
        start, trace, done, reach_radius=scene.target_object.reach_radius, goal_radius=scene.goal_radius,
        movable=scene.target_object.movable, dt=getattr(env, "dt", CTRL_DT),
    )


def summarize_task_arrays(
    start: dict[str, np.ndarray],
    trace: dict[str, np.ndarray],
    done: np.ndarray,
    reach_radius: float,
    goal_radius: float | None = None,
    movable: bool = False,
    dt: float = CTRL_DT,
) -> dict[str, float]:
    """How the scene task went, as means over episodes. Only steps up to each episode's first
    terminal step count. start[k] is (episodes, ...), trace[k] (steps, episodes, ...), done (steps, episodes).

    target_final_in     fraction of episodes that end with the base within reach of the target
    target_near_frac    fraction of the episode the base spends within reach (a proxy for contact)
    final_speed         base speed over the last second, m/s (low = it stopped)
    object_moved        how far the target object moved from where it started, m (movable targets)
    object_goal_*       with a goal: target object to goal distance at the start, closest and end (m),
                        object_goal_reached (fraction ever in the goal), object_in_goal_final (fraction
                        ending in it), object_goal_progress (fraction of the start distance closed;
                        negative = pushed away)
    phase_*             fraction of episodes stuck at each phase (GOAL_PHASES or REACH_PHASES)
    """
    done = np.asarray(done, dtype=bool)
    steps, n = done.shape
    valid = np.concatenate([np.ones_like(done[:1]), np.cumprod(~done[:-1], axis=0).astype(bool)])
    last = np.where(done.any(axis=0), done.argmax(axis=0), steps - 1)  # each episode's final step

    def final(x):
        return x[last, np.arange(n)]

    reached = (trace["dist"] <= reach_radius) & valid
    any_reached = reached.any(axis=0)
    final_in = final(trace["dist"]) <= reach_radius
    tail = int(round(FINAL_S / dt))
    idx = np.arange(steps)[:, None]
    in_tail = valid & (idx > last[None] - tail) & (idx <= last[None])
    out = {
        "target_final_in": float(final_in.mean()),
        "target_near_frac": float((reached.sum(axis=0) / valid.sum(axis=0)).mean()),
        "final_speed": float(((trace["speed"] * in_tail).sum(axis=0) / np.maximum(in_tail.sum(axis=0), 1)).mean()),
    }

    moved = np.zeros(n)
    if movable:
        moved = np.linalg.norm(final(trace["obj_xy"]) - start["obj_xy"], axis=-1)
        out["object_moved"] = float(moved.mean())

    if goal_radius is not None and "goal_dist" in trace:
        g0, g = start["goal_dist"], trace["goal_dist"]
        g_final = final(g)
        in_goal_ever = ((g <= goal_radius) & valid).any(axis=0) | (g0 <= goal_radius)
        in_goal_final = g_final <= goal_radius
        out.update({
            "object_goal_start_dist": float(g0.mean()),
            "object_goal_min_dist": float(np.where(valid, g, np.inf).min(axis=0).mean()),
            "object_goal_final_dist": float(g_final.mean()),
            "object_goal_reached": float(in_goal_ever.mean()),
            "object_in_goal_final": float(in_goal_final.mean()),
            "object_goal_progress": float(((g0 - g_final) / np.maximum(g0, 1e-3)).mean()),
        })
        phase = np.select(
            [in_goal_final, in_goal_ever, ~any_reached & (moved < MOVED), moved < MOVED, g_final >= g0],
            ["delivered", "overshoot", "no_approach", "no_push", "wrong_way"],
            default="short",
        )
        phases = GOAL_PHASES
    else:
        phase = np.select([final_in, any_reached], ["arrived", "left"], default="no_approach")
        phases = REACH_PHASES
    out.update({f"phase_{p}": float((phase == p).mean()) for p in phases})
    return out
