"""The success check: a pass/fail test over rollout metrics, written from the request
alone before any candidate trains, so the target can't move once results come in.

A spec is a list of clauses that must all hold:
    [{"metric": "forward_velocity", "op": "<", "value": -0.4},
     {"metric": "fell", "op": "<", "value": 0.5}]
"""

import json
import math
import operator
from typing import Any

import anthropic

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
OPS = {"<": operator.lt, ">": operator.gt}
MAX_CLAUSES = 6
NO_FALL = {"metric": "fell", "op": "<", "value": 0.5}

SYSTEM = f"""You turn a request for a simulated Unitree Go1 quadruped into a pass/fail
success check over rollout metrics. The check is written before any training and is
never changed afterwards, so it must capture what the user asked for, not what is easy.

The robot is trained with PPO for a few minutes, then rolled out for up to 6 s on a flat
floor. These metrics are measured over the rollout:
{chr(10).join(f"- {name}: {meaning}" for name, (meaning, _, _) in METRICS.items())}

Write 1 to {MAX_CLAUSES} clauses, each "metric op value" with op "<" or ">". All must hold.
- Check the core of the behavior with a margin a decent policy reaches: walking is
  about 0.3 to 1.0 m/s, turning about 1 to 2 rad/s.
- Use the abs_ metrics when the request does not say which way.
- Add a clause against the obvious way to fake it (for example spinning in place:
  abs_forward_velocity < 0.3), but don't demand more than the user asked for.
- Include fell < 0.5 unless the request is to fall or flip.
- If the request can't be measured with these metrics (a gait style, a leg lifted),
  check what can be measured: staying upright and not falling.
"""

SCHEMA = {
    "type": "object",
    "properties": {
        "clauses": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "metric": {"type": "string", "enum": list(METRICS)},
                    "op": {"type": "string", "enum": list(OPS)},
                    "value": {"type": "number"},
                },
                "required": ["metric", "op", "value"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["clauses"],
    "additionalProperties": False,
}


class SpecError(ValueError):
    pass


def validate(clauses: Any) -> list[dict[str, Any]]:
    """Return the spec in canonical form, or raise SpecError saying what is wrong."""
    if not isinstance(clauses, list) or not 1 <= len(clauses) <= MAX_CLAUSES:
        raise SpecError(f"a spec is a list of 1 to {MAX_CLAUSES} clauses")
    out = []
    for c in clauses:
        if not isinstance(c, dict) or set(c) != {"metric", "op", "value"}:
            raise SpecError(f"each clause has exactly metric, op and value: {c!r}")
        metric, op, value = c["metric"], c["op"], c["value"]
        if metric not in METRICS:
            raise SpecError(f"unknown metric {metric!r}; known: {', '.join(METRICS)}")
        if op not in OPS:
            raise SpecError(f"op must be '<' or '>', got {op!r}")
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise SpecError(f"value must be a finite number, got {value!r}")
        _, lo, hi = METRICS[metric]
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
        if not isinstance(actual, (int, float)) or not math.isfinite(actual):
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


class ClaudeChecker:
    def __init__(self, client: anthropic.Anthropic, model: str):
        self.client, self.model = client, model

    def write_spec(self, request: str) -> list[dict[str, Any]]:
        prompt = f"Request: {request}\n\nWrite the success check."
        out = ask_json(self.client, self.model, SYSTEM, [{"type": "text", "text": prompt}], SCHEMA, effort="medium")
        try:
            return validate(out["clauses"])
        except SpecError as exc:  # one retry, told what was wrong
            retry = f"{prompt}\n\nYour last check was rejected: {exc}\nIt was: {json.dumps(out['clauses'])}"
            out = ask_json(self.client, self.model, SYSTEM, [{"type": "text", "text": retry}], SCHEMA, effort="medium")
            return validate(out["clauses"])
