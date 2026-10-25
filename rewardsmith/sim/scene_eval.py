"""Did the trained policy reach the target? Measured over many episodes, not one video.

Runs the seeded, sampled policy in a batch of fresh episodes (objects placed
afresh in each) and reports distances to the target object. The critic reads
these alongside the frames, and they are the headline number for goal tasks.
"""

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from . import outcomes
from .go1 import Go1Env
from .base import ROLLOUT_STEPS


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
