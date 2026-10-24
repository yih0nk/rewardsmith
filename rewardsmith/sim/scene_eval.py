"""Did the trained policy reach the target? Measured over many episodes, not one video.

Runs the seeded, sampled policy in a batch of fresh episodes (objects placed
afresh in each) and reports distances to the target object. The critic reads
these alongside the frames, and they are the headline number for goal tasks.
"""

from collections.abc import Callable
from typing import Any

import jax
import numpy as np

from .go1 import Go1Env
from .base import ROLLOUT_STEPS


def target_metrics(
    env: Go1Env,
    make_policy: Callable[..., Any],
    params: Any,
    num_envs: int = 32,
    steps: int = ROLLOUT_STEPS,
    seed: int = 1,
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
    if target is None:
        return {}
    policy = make_policy(params, deterministic=False)

    def dist(data):
        return env.scene_features(data)["target_dist"]

    def run(rng):
        reset_rng, act_rng = jax.random.split(rng)
        state = jax.vmap(env.reset)(jax.random.split(reset_rng, num_envs))
        start = jax.vmap(dist)(state.data)

        def body(state, key):
            action, _ = policy(state.obs, key)
            state = jax.vmap(env.step)(state, action)
            return state, (jax.vmap(dist)(state.data), state.done)

        _, (dists, done) = jax.lax.scan(body, state, jax.random.split(act_rng, steps))
        return start, dists, done  # (num_envs,), (steps, num_envs), (steps, num_envs)

    start, dists, done = (np.asarray(x) for x in jax.jit(run)(jax.random.PRNGKey(seed)))
    return _summarize(start, dists, done, target.reach_radius, env.dt)


def _summarize(start, dists, done, reach_radius: float, dt: float) -> dict[str, float]:
    """Only observations up to and including each episode's first terminal step count."""
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
