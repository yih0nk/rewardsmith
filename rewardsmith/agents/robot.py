"""Names the request's robot in the agents' prompts.

The prompts were written for the Go1 and say "a simulated Unitree Go1 quadruped". For
another body (sim/bodies) that phrase becomes its title and kind ("Unitree G1 humanoid"),
so the Go1's prompts stay word for word what they were.
"""

import re

from ..sim import bodies

GO1_ROBOT = "Unitree Go1 quadruped"


def robot(body: str | None) -> str:
    """"Unitree Go1 quadruped", "Franka Panda arm", ...: the body's title and kind."""
    b = bodies.get_body(body)
    return f"{b.title} {b.capabilities.kind}"


def kind(body: str | None) -> str:
    return bodies.get_body(body).capabilities.kind


def for_body(text: str, body: str | None) -> str:
    """text with the Go1 named as the body instead, and that body's capability card after it."""
    b = bodies.get_body(body)
    if b.name == bodies.DEFAULT_BODY:
        return text
    return text.replace(GO1_ROBOT, robot(body)).rstrip("\n") + f"\n\nThe robot: {b.capabilities.summary}.\n"


def drop_bullets(text: str, *starts: str) -> str:
    """text without the "- ..." bullets that start with any of starts (and their indented lines)."""
    for start in starts:
        text = re.sub(rf"^- {re.escape(start)}.*?(?=^- |^\S|\Z)", "", text, flags=re.M | re.S)
    return text
