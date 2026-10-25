"""Reads a follow-up note and says whether it changes the scene.

A follow-up ("get behind the ball first", "lift your feet higher", "keep going")
keeps the scene and the success check and revises a reward. A note that changes
the world ("add a box", "use rough ground") needs a new scene: the scene writer
runs again and training starts from scratch, since the old policy saw another world.

One quick call on a fast model, before the editor.
"""

import re

import anthropic

from .claude import ask_json
from .robot import for_body

DEFAULT_MODEL = "claude-haiku-5-5"

SYSTEM = """A user trains a simulated Unitree Go1 quadruped on a behavior. The simulator scene
has a floor (flat or rough) and maybe objects (balls, boxes, ramps, walls, goal markers).
The user watched a result and sent a follow-up note. Decide whether the note asks to change
the SCENE: add, remove, move, resize or swap an object, or change the ground. Notes about
how the robot should move or behave ("go behind the ball first", "faster", "lift your feet",
"keep going", "don't fall") do not change the scene."""

SCHEMA = {
    "type": "object",
    "properties": {
        "changes_scene": {"type": "boolean"},
        "reason": {"type": "string", "description": "One short sentence for the user."},
    },
    "required": ["changes_scene", "reason"],
    "additionalProperties": False,
}


class ClaudeFollowupReader:
    body: str | None = None  # the request\'s robot; the worker sets it per request

    def __init__(self, client: anthropic.Anthropic, model: str = DEFAULT_MODEL):
        self.client, self.model = client, model

    def changes_scene(self, task: str, scene: str, note: str) -> tuple[bool, str]:
        prompt = f"Behavior: {task}\n\nScene now: {scene}\n\nFollow-up note: {note}"
        out = ask_json(
            self.client, self.model, for_body(SYSTEM, self.body), [{"type": "text", "text": prompt}], SCHEMA, effort="low", max_tokens=2000,
        )
        return bool(out["changes_scene"]), out["reason"]


# Offline: a verb that edits the world next to a thing in it.
_EDIT = r"\b(add|put|place|remove|delete|move|replace|swap|make)\b"
_THING = r"\b(ball|box|boxes|crate|ramp|wall|obstacle|goal|target|object|stairs?|step|ground|terrain|floor|hill)s?\b"


class FakeFollowupReader:
    def changes_scene(self, task: str, scene: str, note: str) -> tuple[bool, str]:
        text = note.lower()
        if re.search(_EDIT, text) and re.search(_THING, text) and not re.search(r"\b(behind|toward|towards|to the)\b", text):
            return True, "The note asks for a different scene."
        return False, "The note is about how the robot moves."
