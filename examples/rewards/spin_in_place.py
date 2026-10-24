import jax.numpy as jnp


def reward(state, action, prev_action):
    v = state["base_lin_vel"]
    return {
        "spin": jnp.exp(-jnp.square(state["base_ang_vel"][2] - 1.5) / 0.5),
        "upright": 0.5 * jnp.clip(state["up_z"], 0.0, 1.0),
        "stay_put": -0.5 * (jnp.square(v[0]) + jnp.square(v[1])),
        "smooth": -0.01 * jnp.sum(jnp.square(action - prev_action)),
    }
