"""The 8-behavior benchmark. Each check reads metrics from a rollout.

Thresholds are first guesses. Measure the stock robot at gate G1, then tune.
"""

import operator
from dataclasses import dataclass
from typing import Any, Callable

OPS: dict[str, Callable[[float, float], bool]] = {"<": operator.lt, ">": operator.gt}


@dataclass(frozen=True)
class Check:
    metric: str
    op: str
    threshold: float


@dataclass(frozen=True)
class Behavior:
    slug: str
    prompt: str
    checks: tuple[Check, ...]

    def passes(self, metrics: dict[str, Any]) -> bool:
        for c in self.checks:
            value = metrics.get(c.metric)
            if value is None or not OPS[c.op](float(value), c.threshold):
                return False
        return True


BEHAVIORS: tuple[Behavior, ...] = (
    Behavior("walk_backwards", "walk backwards", (Check("forward_velocity", "<", -0.3),)),
    Behavior("walk_left", "walk sideways to the left",
             (Check("lateral_velocity", ">", 0.3), Check("abs_forward_velocity", "<", 0.15))),
    Behavior("spin", "spin in place", (Check("abs_yaw_rate", ">", 1.0), Check("base_drift_m", "<", 0.5))),
    Behavior("run_fast", "run forward as fast as you can",
             (Check("forward_velocity", ">", 1.0), Check("fall_rate", "<", 0.1))),
    Behavior("crouch_walk", "walk forward while crouching low",
             (Check("height_ratio", "<", 0.7), Check("forward_velocity", ">", 0.2))),
    Behavior("limp", "walk forward without using the front right leg",
             (Check("contact_fr", "<", 0.1), Check("forward_velocity", ">", 0.2))),
    Behavior("pronk", "hop with all four feet leaving the ground together", (Check("airborne_all", ">", 0.2),)),
    Behavior("hind_stand", "stand up on the hind legs", (Check("pitch_deg", ">", 60.0), Check("hold_s", ">", 2.0))),
)
