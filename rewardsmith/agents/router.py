"""Body router: picks the robot for a request when the user left the body on Auto.

Runs once per new task, before the scene writer, and only when more than one body is
registered (with the Go1 alone there is nothing to pick, so nothing runs). It reads each
registered body's capability card (sim/bodies: Capabilities.summary) and returns the body
whose abilities the task needs, with one line saying why, or no body when none can do it.

A pick is {"body": name | None, "reason": str, "suggestion": str}: reason is the board's
line ("the task needs two legs"); for no body, reason says what is missing and suggestion
is the closest request a registered body can do.
"""

import re
from collections.abc import Mapping
from typing import Any

import anthropic

from .claude import ask_json


def cards(bodies: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The capability cards the router reads, one per registered body."""
    out = []
    for name, body in bodies.items():
        c = body.capabilities
        out.append({
            "name": name, "title": body.title, "kind": c.kind, "legs": c.legs, "can_walk": c.can_walk,
            "bipedal": c.bipedal, "can_grasp": c.can_grasp, "reach_m": c.reach_m, "summary": c.summary,
        })
    return out


SYSTEM = """You choose which simulated robot learns a behavior. A user typed a task; another agent
will build the scene and write reinforcement-learning rewards for the robot you pick, and it trains
in MuJoCo for 20 s episodes. You get one capability card per available robot.

Pick the robot whose body the task needs:
- grasping, picking up, lifting, carrying, stacking, placing or reaching for things on a table needs a
  gripper (an arm);
- walking or standing on two legs, waving, gestures with arms, anything "like a person" needs a humanoid;
- walking, trotting, running, turning, climbing, jumping, pushing a ball or box with its body, poses on
  four legs, anything "like a dog" fits a quadruped.
When several robots could do it, prefer the one the task names or implies, then the simplest body
that can do it (a quadruped for plain locomotion). Only pick a robot from the cards.

If no robot on the cards can do the task (it needs a body part, a sense or physics none of them has),
answer body "none", say in reason what is missing, and give in suggestion the closest task one of
the robots can do.

reason is one short line for the user, without the robot's name, e.g. "the task needs two legs",
"picking up the cube needs a gripper", "trotting is a four-legged gait".
"""


def schema(names: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "body": {"type": "string", "enum": [*names, "none"]},
            "reason": {"type": "string"},
            "suggestion": {"type": "string"},
        },
        "required": ["body", "reason", "suggestion"],
        "additionalProperties": False,
    }


def _prompt(request: str, cards_: list[dict[str, Any]]) -> str:
    lines = [f"- {c['name']}: {c['summary']}" for c in cards_]
    return "Robots:\n" + "\n".join(lines) + f"\n\nTask: {request}"


def _clean(out: dict[str, Any], names: list[str]) -> dict[str, Any]:
    body = out.get("body")
    reason = (out.get("reason") or "").strip().rstrip(".")
    return {
        "body": body if body in names else None,
        "reason": reason,
        "suggestion": (out.get("suggestion") or "").strip() if body not in names else "",
    }


class ClaudeRouter:
    def __init__(self, client: anthropic.Anthropic, model: str):
        self.client, self.model = client, model

    def pick(self, request: str, cards_: list[dict[str, Any]]) -> dict[str, Any]:
        names = [c["name"] for c in cards_]
        out = ask_json(self.client, self.model, SYSTEM, [{"type": "text", "text": _prompt(request, cards_)}],
                       schema(names), effort="low", max_tokens=4000)
        return _clean(out, names)


# The fake router's rules: (pattern, what the body must have, reason). First match wins.
GRASP = (r"\b(?:pick(?:s|ing)? (?:\w+ )?up|grab\w*|grasp\w*|lift\w* the|carr(?:y|ies|ying)|stack\w*|plac\w+ the"
         r"|put the|gripper|reach for|hand me)\b")
BIPED = (r"\b(?:two legs|2 legs|bipedal\w*|biped|humanoid|like a (?:person|human)|wave|waving|waves|clap\w*"
         r"|salute\w*|dance like a person)\b")


def _needs(card: dict[str, Any], need: str) -> bool:
    if need == "grasp":
        return card["can_grasp"]
    if need == "biped":
        return card["bipedal"]
    return card["can_walk"]


class FakeRouter:
    """Keyword rules, the offline stand-in for ClaudeRouter."""

    def pick(self, request: str, cards_: list[dict[str, Any]]) -> dict[str, Any]:
        if re.search(GRASP, request, re.I):
            need, reason = "grasp", "the task needs a gripper"
        elif re.search(BIPED, request, re.I):
            need, reason = "biped", "the task needs two legs"
        else:
            need, reason = "walk", "the task is locomotion on legs"
        fits = [c for c in cards_ if _needs(c, need)]
        if need == "walk":  # plain locomotion: the simplest walker, a quadruped before a humanoid
            fits.sort(key=lambda c: c["kind"] != "quadruped")
        if not fits:
            walkers = [c for c in cards_ if c["can_walk"]] or cards_
            return {"body": None, "reason": f"{reason[0].upper()}{reason[1:]} and no available robot has that",
                    "suggestion": "push the ball into the green zone" if walkers and walkers[0]["can_walk"]
                    else "reach the red ball"}
        return {"body": fits[0]["name"], "reason": reason, "suggestion": ""}
