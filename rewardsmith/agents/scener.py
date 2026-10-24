"""Scene writer: turns the user's sentence into a scene spec (sim/scene.py).

Runs once per request, before the reward writer. Its spec decides which
objects exist, so the reward writer gets matching features (ball_dist, ...).
Invalid specs come back with the validator's errors for a fix, like the reward gate.
"""

import json
from typing import Any

import anthropic

from ..sim import scene
from .claude import ask_json

SYSTEM = f"""You set up the world for a robot-learning task. A simulated Unitree Go1 quadruped
(about 0.6 m long, walks up to about 1 m/s, episodes last 20 s) will be trained by reinforcement
learning to do what the user asks. You decide which objects exist around it; another agent then
writes the reward using features of those objects (distance, heading and relative position of each
object; position and velocity of movable ones).

Return a scene spec:
- terrain: "flat", or "rough" (bumps up to 5 cm) only if the user asks for rough, bumpy or uneven ground.
- objects (at most {scene.MAX_OBJECTS}; often 0 or 1): each has
  - name: short lowercase identifier, e.g. "ball", "red_box", "goal". Not {", ".join(sorted(scene.RESERVED_NAMES))}.
  - kind: "ball" (sphere), "box", or "zone" (a coloured disc painted on the floor; nothing collides with it).
  - color: one of {", ".join(scene.COLORS)}. Use the user's colour when they give one.
  - size: ball [radius] {scene.BALL_RADIUS[0]}-{scene.BALL_RADIUS[1]} m (0.1-0.2 is a football);
    zone [radius] {scene.ZONE_RADIUS[0]}-{scene.ZONE_RADIUS[1]} m;
    box [length, width, height] in m, sides {scene.BOX_SIDE[0]}-{scene.BOX_SIDE[1]}, height {scene.BOX_HEIGHT[0]}-{scene.BOX_HEIGHT[1]}
    (a movable box at most {scene.MOVABLE_BOX_SIDE} m per side).
  - distance [low, high]: metres from where the robot starts, 0-{scene.MAX_DISTANCE:g}. Solid objects must start at least
    {scene.ROBOT_CLEARANCE} m + their footprint radius away. Keep things the robot must reach within about 3 m so a
    20 s episode leaves time for the rest of the task.
  - bearing [low, high]: degrees from the robot's starting heading, -180 to 180, positive = to its left.
    Use [-180, 180] when the direction doesn't matter; narrow it for "in front" ([-30, 30]), "behind", "left" etc.
  - yaw [low, high]: degrees the object is turned (boxes); [0, 0] for balls and zones.
  - movable: true only if the robot must push, kick or move it (its feet and body push it). Zones can't be movable.
  - mass: kg, for movable objects 0.05-20 (a football is about 0.4); use 1.0 for fixed ones.
- target: the name of the main object the behavior is about (its features are also given as target_*),
  or "" when there are no objects.

Objects are placed afresh every episode within the ranges, so the policy must learn the skill, not
a spot: prefer ranges over single values. Add only objects the behavior needs; a behavior with no
objects ("walk backwards", "spin in place") gets "objects": [] and "target": "". Low static boxes
(height up to about 0.12 m) work as steps the robot can climb; taller ones are obstacles.
"""

_RANGE = {"type": "array", "items": {"type": "number"}}

SCHEMA = {
    "type": "object",
    "properties": {
        "plan": {"type": "string"},  # one or two sentences: what the task needs; not part of the spec
        "terrain": {"type": "string", "enum": list(scene.TERRAINS)},
        "objects": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "kind": {"type": "string", "enum": list(scene.KINDS)},
                    "color": {"type": "string", "enum": list(scene.COLORS)},
                    "size": _RANGE,
                    "distance": _RANGE,
                    "bearing": _RANGE,
                    "yaw": _RANGE,
                    "movable": {"type": "boolean"},
                    "mass": {"type": "number"},
                },
                "required": list(scene.OBJECT_KEYS),
                "additionalProperties": False,
            },
        },
        "target": {"type": "string"},
    },
    "required": ["plan", "terrain", "objects", "target"],
    "additionalProperties": False,
}


def to_spec(out: dict[str, Any]) -> dict[str, Any]:
    """Claude's answer as a scene spec dict: drop the plan, and "" target means none."""
    return {"terrain": out["terrain"], "objects": out["objects"], "target": out.get("target") or None}


class ClaudeScener:
    def __init__(self, client: anthropic.Anthropic, model: str):
        self.client, self.model = client, model

    def make(self, request: str) -> dict[str, Any]:
        prompt = f"Behavior the user wants: {request}\n\nReturn the scene spec for it."
        return to_spec(ask_json(self.client, self.model, SYSTEM, [{"type": "text", "text": prompt}], SCHEMA,
                                effort="medium"))

    def fix(self, request: str, spec: dict[str, Any], error: str) -> dict[str, Any]:
        prompt = (
            f"Behavior the user wants: {request}\n\n"
            f"This scene spec was rejected:\n{json.dumps(spec)}\n\n{error}\n\n"
            "Return a corrected scene spec. Change only what the errors require."
        )
        return to_spec(ask_json(self.client, self.model, SYSTEM, [{"type": "text", "text": prompt}], SCHEMA,
                                effort="medium"))
