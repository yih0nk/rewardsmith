"""The benchmark cases: each team-plan behavior with a fixed success check.

Each check is a success spec (see agents/checker.py) written here, before any run, from
the metrics the rollout reports today: forward, lateral and yaw velocity (and their
absolute values), base_height, upright and fell. The team plan's original checks in
bench/behaviors.py also need foot contacts, pitch and drift, which the rollout doesn't
measure yet. Where a check below stands in for one of those, `proxy` says how. Two
behaviors can't be checked at all until those metrics exist; they have no spec, and the
report lists them as unmeasured instead of failed.

Thresholds are the team plan's first guesses; tune them against the stock robot.
"""

from dataclasses import dataclass
from typing import Any

from ..sim.go1_spec import STANDING_HEIGHT

NO_FALL = {"metric": "fell", "op": "<", "value": 0.5}


@dataclass(frozen=True)
class Case:
    slug: str
    prompt: str  # the request text, word for word
    spec: tuple[dict[str, Any], ...] | None  # None: no metric can check it yet
    proxy: str = ""  # how the spec differs from the team plan's check, if it does
    reference: bool = False  # not one of the 8 behaviors: the hand-written-reward comparison


def _spec(*clauses: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    return (*clauses, NO_FALL)


def _c(metric: str, op: str, value: float) -> dict[str, Any]:
    return {"metric": metric, "op": op, "value": value}


CASES: tuple[Case, ...] = (
    Case("walk_backwards", "walk backwards", _spec(_c("forward_velocity", "<", -0.3))),
    Case(
        "walk_left",
        "walk sideways to the left",
        _spec(_c("lateral_velocity", ">", 0.3), _c("abs_forward_velocity", "<", 0.15)),
    ),
    Case(
        "spin",
        "spin in place",
        _spec(
            _c("abs_yaw_rate", ">", 1.0),
            _c("abs_forward_velocity", "<", 0.15),
            _c("abs_lateral_velocity", "<", 0.15),
        ),
        proxy="low mean forward and sideways speed stands in for base drift under 0.5 m",
    ),
    Case(
        "run_fast",
        "run forward as fast as you can",
        _spec(_c("forward_velocity", ">", 1.0)),
        proxy="one rollout not falling stands in for a fall rate under 10%",
    ),
    Case(
        "crouch_walk",
        "walk forward while crouching low",
        _spec(_c("base_height", "<", round(0.7 * STANDING_HEIGHT, 3)), _c("forward_velocity", ">", 0.2)),
    ),
    Case("limp", "walk forward without using the front right leg", None, proxy="needs foot contact metrics"),
    Case("pronk", "hop with all four feet leaving the ground together", None, proxy="needs foot contact metrics"),
    Case(
        "hind_stand",
        "stand up on the hind legs",
        _spec(_c("upright", "<", 0.5)),
        proxy="mean upright under 0.5 (body pitched past about 60 degrees on average) "
        "stands in for pitch past 60 degrees held for 2 s",
    ),
    # Compared against examples/rewards/walk_forward.py, the team's hand-written reward,
    # whose regression baseline is in regression/baseline.json.
    Case("walk_forward", "walk forward", _spec(_c("forward_velocity", ">", 0.3)), reference=True),
)

BY_SLUG = {c.slug: c for c in CASES}
