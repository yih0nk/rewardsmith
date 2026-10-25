"""The success check: a pass/fail test over rollout metrics, written from the request
and scene before any candidate trains, so the target can't move once results come in.

A spec is a list of clauses that must all hold:
    [{"metric": "forward_velocity", "op": "<", "value": -0.4},
     {"metric": "fell", "op": "<", "value": 0.5}]
"""

import json
import math
import operator
import re
from typing import Any

import anthropic

from ..sim.scene import EMPTY, SceneSpec
from ..sim import outcomes
from .claude import ask_json

# name: (meaning, lowest sensible threshold, highest). The rollout metrics come from
# sim/render.py; the abs_ ones are derived here, for behaviors with no set direction.
METRICS: dict[str, tuple[str, float, float]] = {
    "forward_velocity": ("mean forward speed in the robot's own frame, m/s; negative is backwards", -3.0, 3.0),
    "lateral_velocity": ("mean sideways speed in the robot's own frame, m/s; positive is to its left", -3.0, 3.0),
    "yaw_rate": ("mean turning rate, rad/s; positive is counter-clockwise seen from above", -10.0, 10.0),
    "abs_forward_velocity": ("absolute value of forward_velocity, m/s", 0.0, 3.0),
    "abs_lateral_velocity": ("absolute value of lateral_velocity, m/s", 0.0, 3.0),
    "abs_yaw_rate": ("absolute value of yaw_rate, rad/s (turning either way)", 0.0, 10.0),
    "base_height": ("mean height of the body above the floor, m; about 0.28 standing", 0.0, 0.6),
    "upright": ("mean z of the body's up axis: 1 level, 0 on its side", -1.0, 1.0),
    "fell": ("1 if the robot flipped over during the rollout, else 0", 0.0, 1.0),
}
TARGET_METRICS = {
    "target_reached": ("fraction of 32 fresh episodes reaching the fixed target before termination; 0 to 1", 0.0, 1.0),
    "target_final_dist": ("mean distance to the target at episode end or termination, metres", 0.0, 20.0),
    "target_fell": ("fraction of those same 32 episodes that fell; 0 to 1", 0.0, 1.0),
}
METRICS.update(TARGET_METRICS)
METRICS["scene_fell"] = ("fraction of fresh evaluation episodes that fell or had invalid telemetry", 0.0, 1.0)
OPS = {"<": operator.lt, ">": operator.gt, "<=": operator.le, ">=": operator.ge}
MAX_CLAUSES = 6
NO_FALL = {"metric": "fell", "op": "<", "value": 0.5}

SYSTEM = f"""You turn a request for a simulated Unitree Go1 quadruped into a pass/fail
success check over rollout metrics. The check is written before any training and is
never changed afterwards, so it must capture what the user asked for, not what is easy.

The robot is trained with PPO, then evaluated for up to 10 s per episode in the supplied scene. These metrics are measured over the rollout:
{chr(10).join(f"- {name}: {meaning}" for name, (meaning, _, _) in METRICS.items())}

Write 1 to {MAX_CLAUSES} clauses, each "metric op value" with op "<", ">", "<=" or ">=". All must hold.
- Check the core of the behavior with a margin a decent policy reaches: walking is
  about 0.3 to 1.0 m/s, turning about 1 to 2 rad/s.
- Use the abs_ metrics when the request does not say which way.
- Add a clause against the obvious way to fake it (for example spinning in place:
  abs_forward_velocity < 0.3), but don't demand more than the user asked for.
- Include fell < 0.5 unless the request is to fall or flip.
- For reaching a fixed object or entering a zone, require target_reached >= 0.8 and
  target_fell <= 0.1 by default (adjust only if the request explicitly demands otherwise).
  These are rates across fresh episodes. Do not require continued forward speed:
  reaching the target and stopping is success. A single nominal video is not enough.
- target_* metrics are available only when the scene has a fixed target.
- Scene outcome metrics (listed with the scene) are success rates across the same episodes.
  Use push:OBJECT:ZONE for pushing, climb:BOX for getting supported on top,
  stand_on:BOX for staying still on top, and avoid:OBJECT for avoiding with progress.
- Combine objectives within each episode: all:reach:goal&avoid:obstacle, not two
  independent aggregate checks that could pass in different episodes.
- Ordered tasks use sequence:reach:ball>climb:step>stand_on:step. Every stage must
  finish after the preceding stage, including a fresh holding interval.
  Use @SECONDS for explicit durations, e.g. stand_still@2 or stand_on:step@3
  (0 < SECONDS <= 10). Honor requested durations instead of shortening them.
  Translate 'A after B' to sequence:B>A and 'A before B' to sequence:A>B.
- Require the task outcome rate >= 0.8 and scene_fell <= 0.1 by default. Do not
  replace pushing/climbing/avoidance/order with just reaching or staying upright.
- If the request needs a measurement outside this scene's event vocabulary, return
  unsupported_reason describing the missing measurement; do not weaken the task.
- For a locomotion request that merely mentions scenery, keep its locomotion checks.
- If the request can't be measured with these metrics (a gait style, a leg lifted),
  check what can be measured: staying upright and not falling.
"""

SCHEMA = {
    "type": "object",
    "properties": {
        "unsupported_reason": {"type": ["string", "null"]},
        "clauses": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "metric": {"type": "string"},
                    "op": {"type": "string", "enum": list(OPS)},
                    "value": {"type": "number"},
                },
                "required": ["metric", "op", "value"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["clauses", "unsupported_reason"],
    "additionalProperties": False,
}


class SpecError(ValueError):
    pass


def metric_definition(metric: str, scene: SceneSpec | None = None):
    if metric in METRICS:
        return METRICS[metric]
    try:
        mode, names = outcomes.parts(metric, scene or EMPTY)
    except ValueError as exc:
        raise SpecError(str(exc)) from exc
    description = (" then " if mode == "sequence" else " and ").join(outcomes.event(name, scene or EMPTY).description for name in names)
    return ("fraction of episodes completing: " + description, 0.0, 1.0)


def validate(clauses: Any, scene: SceneSpec | None = None) -> list[dict[str, Any]]:
    """Return the spec in canonical form, or raise SpecError saying what is wrong."""
    if not isinstance(clauses, list) or not 1 <= len(clauses) <= MAX_CLAUSES:
        raise SpecError(f"a spec is a list of 1 to {MAX_CLAUSES} clauses")
    out = []
    for c in clauses:
        if not isinstance(c, dict) or set(c) != {"metric", "op", "value"}:
            raise SpecError(f"each clause has exactly metric, op and value: {c!r}")
        metric, op, value = c["metric"], c["op"], c["value"]
        if not isinstance(metric, str):
            raise SpecError(f"unknown metric {metric!r}")
        try:
            definition = metric_definition(metric, scene)
        except SpecError as exc:
            raise SpecError(f"unknown metric {metric!r}: {exc}") from exc
        if not isinstance(op, str) or op not in OPS:
            raise SpecError(f"op must be one of {list(OPS)}, got {op!r}")
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise SpecError(f"value must be a finite number, got {value!r}")
        _, lo, hi = definition
        if not lo <= value <= hi:
            raise SpecError(f"{metric} {op} {value} is outside the sensible range {lo} to {hi}")
        out.append({"metric": metric, "op": op, "value": float(value)})
    return out


def evaluate(spec: list[dict[str, Any]], metrics: dict[str, Any]) -> dict[str, Any]:
    """{"pass": all clauses hold, "clauses": [... each with "actual" and "pass"]}.
    A metric the run didn't report fails its clause."""
    values = dict(metrics)
    for name in ("forward_velocity", "lateral_velocity", "yaw_rate"):
        if isinstance(values.get(name), (int, float)):
            values[f"abs_{name}"] = abs(values[name])
    results = []
    for c in spec:
        actual = values.get(c["metric"])
        if isinstance(actual, bool) or not isinstance(actual, (int, float)) or not math.isfinite(actual):
            actual = None  # missing or NaN: the clause fails, and the result stays valid JSON
        ok = actual is not None and OPS[c["op"]](actual, c["value"])
        results.append({**c, "actual": actual, "pass": ok})
    return {"pass": all(r["pass"] for r in results), "clauses": results}


def describe(check: dict[str, Any]) -> str:
    """One line per clause, for prompts and logs."""
    lines = [f"{'PASS' if check['pass'] else 'FAIL'} overall"]
    for c in check["clauses"]:
        lines.append(f"- {c['metric']} {c['op']} {c['value']}: got {c['actual']}, {'pass' if c['pass'] else 'FAIL'}")
    return "\n".join(lines)


REACH_REQUEST = re.compile(r"\b(?:reach(?:es|ing|ed)?|approach(?:es|ing|ed)?|touch(?:es|ing|ed)?|(?:walk|run|go|move|get|navigate|head|travel)(?:\s+\w+){0,2}\s+(?:to|toward|towards)|enter(?:s|ing|ed)?|(?:go|walk|move)\s+(?:inside|into))\b", re.I)
ACTION_WORDS = {
    "forward": r"\b(?:walk|run|move)(?:\s+\w+)?\s+forwards?\b",
    "backward": r"\b(?:walk|run|move)(?:\s+\w+)?\s+backwards?\b",
    "spin": r"\bspin\w*\b",
    "push": r"\b(?:push\w*|kick\w*|dribbl\w*|shov\w*|nudg\w*)\b",
    "climb": r"\b(?:climb\w*|mount\w*|step onto)\b",
    "avoid": r"\b(?:avoid\w*|around\s+(?:the|a|an)\s+\w+|without (?:touching|hitting))\b",
    "stand_still": r"\b(?:stand still|stop|stay still|stand on (?:the |a )?(?:[a-z]+ )?(?:box|step|platform))\b",
}


def requires_reach(request: str) -> bool:
    return bool(REACH_REQUEST.search(request))


def required_actions(request: str) -> set[str]:
    actions = {name for name, pattern in ACTION_WORDS.items() if re.search(pattern, request, re.I)}
    if requires_reach(request) and not actions & {"push", "climb"}:
        actions.add("reach")
    return actions


def _kinds(metric: str, scene: SceneSpec) -> set[str]:
    if metric == "target_reached":
        return {"reach"}
    if metric in METRICS:
        return set()
    _, names = outcomes.parts(metric, scene)
    kinds = {name.split("@")[0].split(":")[0] for name in names}
    if "stand_on" in kinds:
        kinds.update(("stand_still", "climb"))
    return kinds


def validate_for_scene(clauses: Any, request: str, scene: SceneSpec | None) -> list[dict[str, Any]]:
    """Require concrete, scene-valid outcomes and same-episode/order constraints."""
    scene = scene or EMPTY
    spec = validate(clauses, scene)
    target = scene.target_object
    if any(c["metric"] in TARGET_METRICS for c in spec) and (target is None or target.movable):
        raise SpecError("target metrics require a fixed scene target; use named outcomes for movable objects")
    actions = required_actions(request)
    ordered = bool(re.search(r"\b(?:then|after|before)\b", request, re.I))
    if not ordered and actions <= {"forward", "backward", "spin"}:
        actions = set()
    positive = [c for c in spec if c["op"] in (">", ">=") and c["value"] > 0]
    if actions or ordered:
        candidates = [c for c in positive if not ordered or c["metric"].startswith("sequence:")]
        if not any(actions <= _kinds(c["metric"], scene) for c in candidates):
            raise SpecError("The check needs a positive lower bound covering " + ", ".join(sorted(actions)) +
                            (" in a sequence" if ordered else " together in each episode") +
                            "; reaching requires target_reached or reach:OBJECT")
        if not any(c["metric"] in ("target_fell", "scene_fell") and c["op"] in ("<", "<=") and c["value"] < 1 for c in spec):
            raise SpecError("Scene tasks require an upper bound on target_fell or scene_fell across the same episodes")
    durations = [float(value) for value in re.findall(r"(?:for|still)\s+(\d+(?:\.\d+)?)\s*(?:seconds?\b|s\b)", request, re.I)]
    if durations:
        held = []
        for clause in positive:
            if clause["metric"] not in METRICS:
                _, names = outcomes.parts(clause["metric"], scene)
                held.extend(outcomes.event(name, scene).hold_s for name in names)
        remaining = sorted(held)
        for duration in sorted(durations, reverse=True):
            if not remaining or remaining[-1] < duration:
                raise SpecError("The check must measure every requested holding duration")
            remaining.pop()
    return spec


class ClaudeChecker:
    def __init__(self, client: anthropic.Anthropic, model: str):
        self.client, self.model = client, model

    def write_spec(self, request: str, scene: SceneSpec | None = None) -> list[dict[str, Any]]:
        context = scene.summary() if scene else "No scene target: the robot is alone on a flat floor."
        events = "\n".join(f"- {name}: {event.description}" for name, event in outcomes.catalog(scene or EMPTY).items())
        prompt = f"Request: {request}\n\nScene:\n{context}\n\nAvailable episode outcomes:\n{events}\n\nWrite the success check. Use unsupported_reason=null for supported tasks."
        schema = SCHEMA
        for attempt in range(2):
            out = ask_json(self.client, self.model, SYSTEM, [{"type": "text", "text": prompt}], schema, effort="medium")
            if out.get("unsupported_reason"):
                raise SpecError(f"Unsupported task: {out['unsupported_reason']}")
            try:
                return validate_for_scene(out.get("clauses"), request, scene)
            except SpecError as exc:
                if attempt:
                    raise
                prompt += f"\n\nYour last check was rejected: {exc}\nIt was: {json.dumps(out.get('clauses'))}"
        raise AssertionError("unreachable")
