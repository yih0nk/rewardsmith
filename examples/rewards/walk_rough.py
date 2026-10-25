import jax.numpy as jnp

# Scene: examples/scenes/rough.json (bumps up to 5 cm, no objects).
# Recipe for walking on rough ground: the flat-ground walking terms, plus foot clearance so
# the feet clear the bumps, and an attitude term instead of a height target (base_height is
# world z, which moves with the bumps).


def reward(state, action, prev_action):
    v = state["base_lin_vel"]
    swing = 1.0 - state["foot_contact"]
    return {
        "forward": jnp.exp(-jnp.square(v[0] - 0.5) / 0.1),
        "upright": 0.5 * jnp.clip(state["up_z"], 0.0, 1.0),
        "level": -1.0 * jnp.sum(jnp.square(state["gravity"][:2])),
        "clearance": 0.3 * jnp.sum(swing * jnp.clip(state["feet_height"] - state["base_pos"][2] + 0.22, 0.0, 0.08)),
        "no_sideways": -0.5 * jnp.square(v[1]),
        "no_spin": -0.2 * jnp.square(state["base_ang_vel"][2]),
        "smooth": -0.01 * jnp.sum(jnp.square(action - prev_action)),
    }
