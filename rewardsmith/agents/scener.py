"""Scene writer: turns the user's sentence into a scene spec (sim/scene.py).

Runs once per request, before the reward writer. Its spec decides which
objects and structures exist, so the reward writer gets matching features (ball_dist, ...).
Invalid specs come back with the validator's and the scene gate's errors for a fix,
like the reward gate.

The same call classifies the request first: doable, approximate (with the
simplification stated for the user) or refuse (with a plain reason and the closest
doable prompt), so impossible requests end before any GPU time.
"""

import json
from typing import Any

import anthropic

from ..sim import scene
from .claude import ask_json

SYSTEM = f"""You set up the world for a robot-learning task. A simulated Unitree Go1 quadruped
(about 0.6 m long and 0.3 m tall, legs about 0.2 m, walks up to about 1 m/s, episodes last 20 s) will be
trained by reinforcement learning to do what the user asks. First decide whether the request can be done
at all, then build the world: which objects and structures exist around the robot. Another agent then
writes the reward using features of the relevant ones (distance, heading and relative position; position
and velocity of movable ones; ground height under the feet when there is terrain).

1. feasibility: classify the request.
- "doable": this robot can learn it in this simulator.
- "approximate": doable only with a stated simplification. Say exactly what you simplify in
  `simplification` (shown to the user), e.g. "a coloured zone stands in for the kitchen", "a low
  capsule stands in for the log", "the 'friend' is a ball the robot follows".
- "refuse": it cannot be done, so training would only produce a misleading result. Refuse when it needs
  - a body the Go1 doesn't have: grasping, picking up or carrying with a hand or mouth, throwing,
    flying or jumping over things taller than about 0.15 m, walking upright on two legs for long, talking;
  - physics the simulator doesn't have: cloth, ropes, liquids, sand, deformable or breakable things,
    fire, magnets, other animals or people that act on their own;
  - something that isn't a physical behavior at all: emotions, personality, "be friendly", "be happy",
    moods, opinions;
  - choosing between things by appearance the robot can't sense: it senses where each relevant object
    is, never its colour, texture or label, so "go to whichever ball is red" (when the colour changes per
    episode) or "find the cat in the picture" can't be learned. A fixed named object of a fixed colour is fine.
  Standing up on the hind legs briefly (rearing up, begging) IS doable; walking on two legs is not.
  Climbing low stairs (steps up to about 0.1 m), walking up gentle ramps (up to about 15 degrees),
  weaving between poles, dodging a rolled ball, pushing light objects, stepping over a low bar are doable.
  `reason`: one plain sentence for the user. `suggestion`: for refuse and approximate, the closest
  request that is doable (e.g. "rear up on your hind legs for 2 seconds"); else "".
  For refuse, still return an empty scene (no objects or structures).

2. terrain: "flat", or "rough" (bumps up to 5 cm) only if the user asks for rough, bumpy or uneven ground.

3. objects (at most {scene.MAX_OBJECTS}): single bodies, each with
  - name: short lowercase identifier, e.g. "ball", "red_box", "pole1". Not {", ".join(sorted(scene.RESERVED_NAMES))}.
  - kind: "ball" (sphere), "box", "capsule" (a rounded cylinder: upright it is a pole or post; tilted
    90 degrees it is a bar or log), or "zone" (a coloured disc painted on the floor; nothing collides with it).
  - color: one of {", ".join(scene.COLORS)}. Use the user's colour when they give one.
  - size: ball [radius] {scene.BALL_RADIUS[0]}-{scene.BALL_RADIUS[1]} m (0.1-0.2 is a football);
    zone [radius] {scene.ZONE_RADIUS[0]}-{scene.ZONE_RADIUS[1]} m;
    box [length, width, height] in m, sides {scene.BOX_SIDE[0]}-{scene.BOX_SIDE[1]}, height {scene.BOX_HEIGHT[0]}-{scene.BOX_HEIGHT[1]}
    (a movable box at most {scene.MOVABLE_BOX_SIDE} m per side);
    capsule [radius, length] radius {scene.CAPSULE_RADIUS[0]}-{scene.CAPSULE_RADIUS[1]}, length {scene.CAPSULE_LENGTH[0]}-{scene.CAPSULE_LENGTH[1]}
    (the straight part; total length adds 2 radii). A pole is about [0.05, 1.0].
  - distance [low, high]: metres from where the robot starts, 0-{scene.MAX_DISTANCE:g}. Solid objects must start at least
    {scene.ROBOT_CLEARANCE} m + their footprint radius away. Keep things the robot must reach within about 3 m.
  - bearing [low, high]: degrees from the robot's starting heading, -180 to 180, positive = to its left.
    Use [-180, 180] when the direction doesn't matter; narrow it for "in front" ([-30, 30]), "behind", "left" etc.
    Objects that must not overlap need ranges that keep them apart (e.g. poles in a row: distances 1.5, 2.5, 3.5
    with bearings alternating [8, 12] and [-12, -8]).
  - yaw [low, high]: degrees the object is turned; [0, 0] for balls and zones.
  - tilt [roll, pitch]: fixed degrees, [0, 0] usually; [90, 0] lays a capsule across the robot's path.
  - elevation: metres from the floor to the object's lowest point, for fixed objects only (a hurdle bar at 0.1); else 0.
  - movable: true only if the robot must push or kick it, or it is launched. Zones can't be movable.
  - mass: kg, for movable objects 0.05-20 (a football is about 0.4); use 1.0 for fixed ones.
  - speed [low, high]: m/s, for movable objects launched at the robot's start position at reset
    (e.g. "dodge the rolling ball": a ball 4-6 m away at speed [1.5, 2.5]); [0, 0] otherwise.
  - aim [low, high]: degrees the launch is off the line to the robot's start; [0, 0] aims straight at it.
  - relevant: true when the behavior is about it (it gets features and the robot senses it); false for scenery.

4. structures (at most {scene.MAX_STRUCTURES}): parametric terrain built from several pieces, each with
  name, kind, color, distance (to its near edge), bearing, yaw, relevant, and params ([{{"key": ..., "value": ...}}];
  omit a key to use its default). It faces the robot: with yaw [0, 0], a robot walking straight at it climbs
  the stairs or ramp, or meets the wall broadside. Kinds and params (default, low-high):
{chr(10).join(f"  - {kind}: " + ", ".join(f"{k} {d:g} ({lo:g}-{hi:g})" for k, (d, lo, hi) in table.items()) + f". {scene.STRUCTURE_HELP[kind]}" for kind, table in scene.STRUCTURE_PARAMS.items())}
  The Go1 manages steps up to about 0.1 m rise (0.05-0.08 is comfortable) and ramps up to about 15 degrees.
  A structure's features describe its anchor: the top landing of stairs and ramps, the far platform of a gap,
  a wall's centre, a field's far edge. Use a field for "walk through the obstacles", a wall with low height
  for a hurdle, stairs for "climb the stairs", a ramp for "walk up the ramp".
  At most {scene.MAX_GEOMS} pieces in total (each stair step and each field obstacle is one piece).

5. target: the name of the main object or structure the behavior is about (its features are also given as
  target_*), or "" when there is none. goal: for "move / push / kick the target to somewhere" tasks, the name
  of the place it must end up (usually a zone; the target must be a movable object), else "".

For a move-it-somewhere task, lay it out so a straight push can succeed in one 20 s episode: the target
about 1-1.5 m from the robot, and the goal beyond it as seen from the robot's start (a similar bearing
range, about 1.5-2 m farther out), with a goal zone radius of at least 0.4 m.

Objects are placed afresh every episode within the ranges, so the policy must learn the skill, not
a spot: prefer ranges over single values. Add only what the behavior needs; a behavior with no
objects ("walk backwards", "spin in place") gets no objects or structures and "target": "".
Your spec is checked in physics (pieces overlapping each other or the robot's start are rejected); fix what
the errors name.
"""

_RANGE = {"type": "array", "items": {"type": "number"}}
_PARAM_KEYS = sorted({k for table in scene.STRUCTURE_PARAMS.values() for k in table})

SCHEMA = {
    "type": "object",
    "properties": {
        "plan": {"type": "string"},  # one or two sentences: what the task needs; not part of the spec
        "feasibility": {
            "type": "object",
            "properties": {
                "verdict": {"type": "string", "enum": ["doable", "approximate", "refuse"]},
                "reason": {"type": "string"},
                "simplification": {"type": "string"},
                "suggestion": {"type": "string"},
            },
            "required": ["verdict", "reason", "simplification", "suggestion"],
            "additionalProperties": False,
        },
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
                    "tilt": _RANGE,
                    "elevation": {"type": "number"},
                    "movable": {"type": "boolean"},
                    "mass": {"type": "number"},
                    "speed": _RANGE,
                    "aim": _RANGE,
                    "relevant": {"type": "boolean"},
                },
                "required": list(scene.OBJECT_KEYS),
                "additionalProperties": False,
            },
        },
        "structures": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "kind": {"type": "string", "enum": list(scene.STRUCTURE_KINDS)},
                    "color": {"type": "string", "enum": list(scene.COLORS)},
                    "distance": _RANGE,
                    "bearing": _RANGE,
                    "yaw": _RANGE,
                    "relevant": {"type": "boolean"},
                    "params": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {"key": {"type": "string", "enum": _PARAM_KEYS}, "value": {"type": "number"}},
                            "required": ["key", "value"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": list(scene.STRUCTURE_KEYS),
                "additionalProperties": False,
            },
        },
        "target": {"type": "string"},
        "goal": {"type": "string"},
    },
    "required": ["plan", "feasibility", "terrain", "objects", "structures", "target", "goal"],
    "additionalProperties": False,
}


def to_spec(out: dict[str, Any]) -> dict[str, Any]:
    """Claude's answer as a scene spec dict: drop the plan, "" target or goal means none, params
    become a dict. The feasibility verdict rides along under FEASIBILITY for the orchestrator."""
    structures = [{**st, "params": {p["key"]: p["value"] for p in st.get("params") or []}}
                  for st in out.get("structures") or []]
    spec = {"terrain": out["terrain"], "objects": out["objects"], "structures": structures,
            "target": out.get("target") or None, "goal": out.get("goal") or None}
    if out.get("feasibility"):
        spec[FEASIBILITY] = out["feasibility"]
    return spec


FEASIBILITY = "feasibility"


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
            f"This scene spec was rejected:\n{json.dumps({k: v for k, v in spec.items() if k != FEASIBILITY})}\n\n{error}\n\n"
            "Return a corrected scene spec. Change only what the errors require."
        )
        return to_spec(ask_json(self.client, self.model, SYSTEM, [{"type": "text", "text": prompt}], SCHEMA,
                                effort="medium"))
