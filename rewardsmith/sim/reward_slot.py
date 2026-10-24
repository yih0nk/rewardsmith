"""Load a generated reward module into a callable.

The source runs with a small namespace (jax, jnp, math). This is a guard against
mistakes, not a security sandbox: only run sources our own writer produced.
"""

import math
from collections.abc import Callable
from typing import Any


def load_reward(source: str, extra_globals: dict[str, Any] | None = None) -> Callable[..., dict]:
    namespace: dict[str, Any] = {"math": math, "__name__": "reward_module"}
    try:
        import jax
        import jax.numpy as jnp

        namespace.update(jax=jax, jnp=jnp)
    except ImportError:
        pass
    if extra_globals:
        namespace.update(extra_globals)

    exec(compile(source, "reward.py", "exec"), namespace)
    fn = namespace.get("reward")
    if not callable(fn):
        raise ValueError("reward.py must define a function named reward(state, action, prev_action)")
    return fn
