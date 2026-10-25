"""The robots the loop can train, by name (see docs/bodies.md).

Each body module calls register() on import; the import lines at the bottom
are the registry. Body modules must stay cheap to import (no jax/mujoco at
module level), since the server and the writer read describe() without a GPU.
"""

from .base import Body, Capabilities, RewardError  # noqa: F401

DEFAULT_BODY = "go1"
BODIES: dict[str, Body] = {}


def register(body: Body) -> Body:
    if body.name in BODIES:
        raise ValueError(f"body {body.name!r} is registered twice")
    BODIES[body.name] = body
    return body


def get_body(name: str | None = None) -> Body:
    """The body called name (None: the default, the Go1)."""
    name = name or DEFAULT_BODY
    try:
        return BODIES[name]
    except KeyError:
        raise ValueError(f"unknown body {name!r}: use one of {sorted(BODIES)}") from None


# One line per body.
from . import go1  # noqa: E402, F401
