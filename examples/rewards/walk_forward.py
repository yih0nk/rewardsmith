import jax.numpy as jnp


def reward(state, action, prev_action):
    v = state["base_lin_vel"]
    return {
        "forward": jnp.exp(-jnp.square(v[0] - 0.6) / 0.1),
        "upright": 0.5 * jnp.clip(state["up_z"], 0.0, 1.0),
        "no_sideways": -0.5 * jnp.square(v[1]),
        "no_spin": -0.2 * jnp.square(state["base_ang_vel"][2]),
        "smooth": -0.01 * jnp.sum(jnp.square(action - prev_action)),
    }
