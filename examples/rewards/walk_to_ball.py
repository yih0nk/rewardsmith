import jax.numpy as jnp

# Scene: examples/scenes/red_ball.json (a fixed red ball 1.5-3 m away, any direction).


def reward(state, action, prev_action):
    dist = state["target_dist"]
    rel = state["target_rel_pos"]
    v = state["base_lin_vel"]
    # 1 once the base is within 0.45 m of the ball's centre (nose on the ball), falling off fast beyond.
    arrived = jnp.exp(-jnp.square(jnp.maximum(dist - 0.45, 0.0)) / 0.02)
    # Speed toward the ball, in the robot's frame, capped at a brisk walk.
    toward = jnp.dot(v[:2], rel[:2] / jnp.maximum(dist, 1e-3))
    return {
        "approach": 1.5 * jnp.clip(toward, -0.5, 0.8) * (1.0 - arrived),
        "arrive": 2.0 * arrived,
        "face": 0.3 * jnp.cos(state["target_heading_err"]),
        "settle": -0.5 * arrived * jnp.sum(jnp.square(v[:2])),
        "stand_tall": 0.5 * jnp.exp(-jnp.square(state["base_height"] - 0.28) / 0.002),
        "upright": 0.5 * jnp.clip(state["up_z"], 0.0, 1.0),
        "smooth": -0.01 * jnp.sum(jnp.square(action - prev_action)),
    }
