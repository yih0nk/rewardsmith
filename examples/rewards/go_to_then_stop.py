import jax.numpy as jnp

# Scene: examples/scenes/blue_box.json (a fixed box 1.5-3 m away, ahead or to the side).
# Recipe for "go to X and stop beside it": pay for speed toward X while far, a bounded
# arrival bonus once in reach, and a bonus for holding still once there. Still means no
# turning too: a policy that stops walking but keeps shuffling round in place hasn't stopped.
# No term pays for standing still away from X, so the only way to collect is to get there.

REACH = 0.6  # target_dist below this is "beside it", from the env notes (reach radius minus a little)


def reward(state, action, prev_action):
    dist = state["target_dist"]
    rel = state["target_rel_pos"][:2]
    v = state["base_lin_vel"]
    arrived = jnp.exp(-jnp.square(jnp.maximum(dist - REACH, 0.0)) / 0.02)
    toward = jnp.dot(v[:2], rel / jnp.maximum(dist, 1e-3))
    return {
        "approach": 1.5 * jnp.clip(toward, -0.5, 0.8) * (1.0 - arrived),
        "arrive": 2.0 * arrived,
        "face": 0.3 * jnp.cos(state["target_heading_err"]),
        "still": 1.5 * arrived * jnp.exp(-jnp.sum(jnp.square(v[:2])) / 0.01 - jnp.square(state["base_ang_vel"][2]) / 0.02),
        "stand_tall": 0.5 * jnp.exp(-jnp.square(state["base_height"] - 0.28) / 0.002),
        "upright": 0.5 * jnp.clip(state["up_z"], 0.0, 1.0),
        "no_bounce": -0.5 * jnp.square(v[2]),
        "smooth": -0.01 * jnp.sum(jnp.square(action - prev_action)),
    }
