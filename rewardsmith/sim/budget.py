"""Per-request training plan: how many steps each candidate trains, and which stock terms it keeps.

Keyword rules plus the scene, no Claude call, so it is instant and repeatable:

  simple   one skill on the open floor (walk backwards, spin, trot, stand on hind legs):
           the pretrained walker already has the balance, the reward only steers it
  object   the scene has objects to reach, face or avoid
  complex  a movable object with a goal or zone, or several steps in a row ("then", "after")

and a family: "pose" (stand on hind legs, sit, jump, roll over, ...) fine-tunes the walker and
drops the stock terms that hold the body level (go1_spec.STOCK_PROFILES); "gait" steers the
walker with velocity commands (playground.PlaygroundSimulator.plan_request).
The step counts were measured on the 4090 (docs/warm-start.md). Steering the walker, the
regression's walk forward / walk backwards / spin rewards matched their 10M-step cold
baselines at 5M (walk backwards was at 28.9 of 29.5 after 1.3M); on the live ball-to-goal
rewards the steered eval reward was still climbing at 10M, so complex tasks get 20M.
"""

import re
from dataclasses import dataclass

from .scene import SceneSpec

BUDGETS: dict[str, int] = {
    "simple": 5_000_000,
    "object": 10_000_000,
    "complex": 20_000_000,
}

POSE = re.compile(
    r"hind legs|back legs|two legs|rear(?:s|ing)? up|stand(?:s|ing)? (?:up|tall)|\bsit|lie down|lay down|"
    r"handstand|headstand|\bjump|\bhop|\bleap|roll(?:s|ing)? over|\bflip|\bbow|\bbeg|danc|crouch|crawl|"
    r"\bwave|shake|stretch|play dead|wiggle",
    re.IGNORECASE,
)
STEPS = re.compile(r"\bthen\b|\bafter(?:wards)?\b|\bbefore\b|\bfinally\b|\bnext\b|\bfirst\b.*\bthen\b", re.IGNORECASE)
MOVE_IT = re.compile(r"\b(?:push|move|shove|kick|carry|bring|deliver|roll|drag|nudge|herd|dribble)", re.IGNORECASE)


@dataclass(frozen=True)
class Plan:
    kind: str  # simple | object | complex
    family: str  # gait | pose (a go1_spec.STOCK_PROFILES key)
    steps: int
    why: str

    def summary(self) -> str:
        return f"Training plan: {self.kind} task ({self.why}), {self.steps / 1e6:g}M steps per candidate, {self.family} family"


def plan(text: str, scene: SceneSpec) -> Plan:
    family = "pose" if POSE.search(text) else "gait"
    movable = [o for o in scene.objects if o.movable]
    zones = [o for o in scene.objects if o.kind == "zone"]
    goal = getattr(scene, "goal", None)
    if movable and (zones or goal or MOVE_IT.search(text)):
        kind, why = "complex", "moves an object to a place"
    elif STEPS.search(text):
        kind, why = "complex", "several steps"
    elif scene.objects:
        kind, why = "object", f"scene has {', '.join(o.name for o in scene.objects)}"
    elif scene.terrain == "rough":
        kind, why = "object", "rough ground"
    else:
        kind, why = "simple", "one skill on open floor"
    return Plan(kind, family, BUDGETS[kind], why)
