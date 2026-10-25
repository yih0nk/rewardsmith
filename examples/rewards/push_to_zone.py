import jax.numpy as jnp

# Scene: examples/scenes/ball_to_zone.json (a movable ball 1-1.5 m ahead, a goal zone beyond it).
# Recipe for "move the object to the goal": walk to the spot behind the object (on the far side
# from the goal), then push. The main term pays for the object closing on the goal, the per-step
# decrease of the object-to-goal distance (potential-based progress, so parking pays nothing).

BALL_R = 0.15  # the object's radius / half-length, from the env notes
GOAL_R = 0.5  # target_to_goal_dist below this is "in the goal", from the env notes


def reward(state, action, prev_action):
    v = state["base_lin_vel"][:2]
    ball = state["target_rel_pos"][:2]  # robot -> ball, robot heading frame
    to_goal = state["target_to_goal_rel"][:2]  # ball -> goal, same frame
    goal_dist = state["target_to_goal_dist"]
    u = to_goal / jnp.maximum(jnp.linalg.norm(to_goal), 1e-3)
    ball_dir = ball / jnp.maximum(jnp.linalg.norm(ball), 1e-3)
    # 1 when the robot is straight behind the ball, looking down the ball -> goal line.
    aligned = jnp.clip(jnp.dot(ball_dir, u), 0.0, 1.0)
    # Where to walk: the push spot behind the ball until lined up, then the ball itself.
    behind = ball - (BALL_R + 0.3) * u
    aim = jnp.where(aligned > 0.9, ball, behind)
    aim_dir = aim / jnp.maximum(jnp.linalg.norm(aim), 1e-3)
    in_goal = jnp.exp(-jnp.square(jnp.maximum(goal_dist - 0.5 * GOAL_R, 0.0)) / 0.02)
    return {
        "approach": 1.0 * jnp.clip(jnp.dot(v, aim_dir), -0.3, 0.8) * (1.0 - in_goal),
        "progress": 4.0 * jnp.clip(state["target_to_goal_speed"], -0.5, 1.0),
        "aligned": 0.3 * aligned * (1.0 - in_goal),
        "face": 0.2 * jnp.cos(state["target_heading_err"]),
        "in_goal": 2.0 * in_goal,
        "stand_tall": 0.5 * jnp.exp(-jnp.square(state["base_height"] - 0.28) / 0.002),
        "upright": 0.5 * jnp.clip(state["up_z"], 0.0, 1.0),
        "no_bounce": -0.5 * jnp.square(state["base_lin_vel"][2]),
        "no_wobble": -0.05 * jnp.sum(jnp.square(state["base_ang_vel"][:2])),
        "smooth": -0.01 * jnp.sum(jnp.square(action - prev_action)),
    }
