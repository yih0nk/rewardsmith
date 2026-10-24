import jax.numpy as jnp

# Scene: examples/scenes/red_ball.json (a fixed red ball 1.5-3 m away, any direction).


def reward(state, action, prev_action):
    dist = state["target_dist"]
    rel = state["target_rel_pos"]
    v = state["base_lin_vel"]
    # 1 once the base is within half a metre of the ball's centre (touching it), fading out beyond.
    arrived = jnp.exp(-jnp.square(jnp.maximum(dist - 0.5, 0.0)) / 0.1)
    # Speed toward the ball, in the robot's frame, capped at a brisk walk.
    toward = jnp.dot(v[:2], rel[:2] / jnp.maximum(dist, 1e-3))
    return {
        "approach": 1.5 * jnp.clip(toward, -0.5, 0.8) * (1.0 - arrived),
        "arrive": 2.0 * arrived,
        "face": 0.3 * jnp.cos(state["target_heading_err"]),
        "settle": -0.5 * arrived * jnp.sum(jnp.square(v[:2])),
        "upright": 0.5 * jnp.clip(state["up_z"], 0.0, 1.0),
        "smooth": -0.01 * jnp.sum(jnp.square(action - prev_action)),
    }
