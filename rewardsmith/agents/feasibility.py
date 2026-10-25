"""Can this robot learn the request at all? Keyword rules, for the fake scene writer.

The real scene writer (scener.py) classifies every request with Claude in the same call
that builds the scene. These rules are its offline stand-in, and they only catch the
plain cases: a request needing a body or physics the Go1 in MuJoCo doesn't have, or
one that isn't a physical behavior, is refused; one that names a place or game the
scene can only stand in for is approximated; everything else is doable.

A verdict is {"verdict": "doable" | "approximate" | "refuse", "reason", "simplification",
"suggestion"}, the scene writer's schema.
"""

import re
from typing import Any

# (pattern, reason, closest doable request)
REFUSE: tuple[tuple[str, str, str], ...] = (
    (r"\b(?:pick(?:s|ing)? up|grab\w*|grasp\w*|carr(?:y|ies|ying)|throw\w*|catch\w*|fold\w*|open the|hold (?:the|a|an))\b",
     "the Go1 has no hand or gripper, so it can't grasp, carry, throw or fold things",
     "push the ball into the green zone"),
    (r"\b(?:fly|flies|flying|hover\w*|teleport\w*)\b", "the robot can't leave the ground for more than a short hop",
     "walk up the ramp and stop at the top"),
    (r"\b(?:walk\w*|run\w*|go|move|cross\w*|travel\w*)\b.{0,40}\bon (?:two|its|your|the) (?:hind |back )?legs\b"
     r"|\b(?:upright like a (?:person|human)|bipedal\w*)\b",
     "walking on two legs is beyond a quadruped's balance; it can only rear up briefly",
     "stand on your hind legs for 2 seconds"),
    (r"\b(?:swim\w*|water|pool|liquid\w*|pour\w*|splash\w*|cloth|towel|rope|sand|mud|fire|smoke)\b",
     "the simulator has only rigid bodies: no liquids, cloth, rope or sand",
     "walk across the rough ground"),
    (r"\b(?:be (?:friendly|happy|sad|nice|kind|polite|brave|cute)|feel\w*|emotion\w*|happy|sad|angry|love\w*|mood)\b",
     "a mood or personality isn't a physical behavior that can be measured",
     "wag around: spin in place once, then sit still"),
    (r"\b(?:bark\w*|talk\w*|speak\w*|sing\w*|say\w*)\b", "the robot has no voice", "walk to the red ball"),
    (r"\b(?:whichever|which ever|recogni[sz]e\w*|read (?:the|a)|in the picture|shuffl\w*|random colou?rs?)\b",
     "the robot senses where each object is but never its colour or label, so it can't choose by appearance",
     "walk to the red ball"),
)
APPROXIMATE: tuple[tuple[str, str], ...] = (
    (r"\b(?:kitchen|room|house|door|garden|yard|park|street|home)\b", "a coloured zone stands in for the place"),
    (r"\b(?:mountain|hill|slope)\b", "a ramp stands in for the hill"),
    (r"\b(?:bridge|river|ledge)\b", "two platforms with a gap stand in for the bridge"),
    (r"\b(?:play\w*|fetch\w*|game)\b", "a movable ball to push around stands in for playing"),
    (r"\b(?:patrol\w*|guard\w*)\b", "walking between two zones stands in for patrolling"),
    (r"\b(?:follow\w*|chase\w*)\b", "a ball rolled past the robot stands in for the thing to follow"),
    (r"\blike a (?:cat|dog|horse|tiger|lion)\b", "a plain walk stands in for the animal's style"),
)


def classify(request: str) -> dict[str, Any]:
    for pattern, reason, suggestion in REFUSE:
        if re.search(pattern, request, re.I):
            return {"verdict": "refuse", "reason": f"Can't train this: {reason}.", "simplification": "",
                    "suggestion": suggestion}
    for pattern, simplification in APPROXIMATE:
        if re.search(pattern, request, re.I):
            return {"verdict": "approximate", "reason": "Doable with a simplification.",
                    "simplification": simplification[0].upper() + simplification[1:] + ".", "suggestion": ""}
    return {"verdict": "doable", "reason": "", "simplification": "", "suggestion": ""}
