"""Checks for a generated Go1 reward module before it goes into the env."""

from collections.abc import Callable, Mapping
from typing import Any

import jax
import jax.numpy as jnp

from .go1_spec import FEATURES

RewardFn = Callable[[dict[str, jax.Array], jax.Array, jax.Array], Mapping[str, Any]]


class RewardError(ValueError):
    """The reward module is unusable. The message is meant for the reward writer."""


def dummy_state() -> dict[str, jax.Array]:
    """A zero-filled state with the real keys and shapes, for checking a reward eagerly."""
    state = {f.name: jnp.zeros(f.shape, dtype=jnp.float32) for f in FEATURES}
    state["gravity"] = jnp.array([0.0, 0.0, -1.0])
    state["up_z"] = jnp.float32(1.0)
    return state


def discover_terms(reward_fn: RewardFn) -> tuple[str, ...]:
    """Call the reward once on a dummy state and return its term names, sorted."""
    zeros = jnp.zeros(12, dtype=jnp.float32)
    try:
        terms = reward_fn(dummy_state(), zeros, zeros)
    except Exception as exc:
        raise RewardError(f"reward() raised on a standing robot: {type(exc).__name__}: {exc}") from exc
    if not isinstance(terms, Mapping) or not terms:
        raise RewardError("reward() must return a non-empty dict of named terms")
    for name, value in terms.items():
        if not isinstance(name, str) or not name.isidentifier():
            raise RewardError(f"term name {name!r} must be a plain identifier like 'backward_velocity'")
        size = jnp.size(jnp.asarray(value))
        if size != 1:
            raise RewardError(f"term {name!r} must be a scalar, got {size} values")
    return tuple(sorted(terms))
