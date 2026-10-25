"""The success check: a pass/fail test over rollout metrics, written from the request
and scene before any candidate trains, so the target can't move once results come in.

A spec is a list of clauses that must all hold:
    [{"metric": "forward_velocity", "op": "<", "value": -0.4},
     {"metric": "fell", "op": "<", "value": 0.5}]

When no metric in the vocabulary measures the task (stairs climbed, poles weaved, a
pose held, a ball dodged), the checker also writes check(traj), a small JAX function
over each evaluation episode (sim/check_code.py), and its clauses name the results
as check:<key>. The code passes a gate first (AST guard, jitted dry run), and a check
that holds for a robot just standing where it starts is rejected unless standing
still is the task. Every check needs at least one task-specific clause: not falling
and staying upright never count as the task.
"""

import json
import math
import operator
import re
from typing import Any

import anthropic

from dataclasses import dataclass, field

from ..sim.go1_spec import POSE_METRICS
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
METRICS.update(POSE_METRICS)  # feet off the floor, held time, pitch, roll: for pose tasks
OPS = {"<": operator.lt, ">": operator.gt, "<=": operator.le, ">=": operator.ge}
MAX_CLAUSES = 6
NO_FALL = {"metric": "fell", "op": "<", "value": 0.5}
CHECK_PREFIX = "check:"  # metrics computed by the checker's own check(traj) code
# Clauses that never show the task got done on their own.
GENERIC_METRICS = frozenset({"fell", "scene_fell", "target_fell", "upright"})
CODE_RANGE = (-1e6, 1e6)  # a float result's sensible range; bools are fractions, 0 to 1


@dataclass
class Check:
    """A validated success check: clauses, and the code behind any check:<key> clause."""

    clauses: list[dict[str, Any]]
    code: str | None = None
    keys: dict[str, str] = field(default_factory=dict)  # check_code.CheckGate.keys
    standing: dict[str, float] = field(default_factory=dict)  # each result on a robot standing still

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
- Include fell < 0.5 unless the request is to fall or flip. With a scene, use only
  scene_fell (or target_fell) instead: those count falls over the same fresh episodes as the
  task, and a second fall clause on the one video only disagrees with them.
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
- Pose requests (stand on the hind legs, lift the front paws, a handstand, sit up) must
  check the pose itself with the pose metrics, e.g. standing on the hind legs:
  front_feet_off >= 0.6, pitch_deg >= 45 and base_height >= 0.3, plus front_off_held_s >= the
  seconds asked for when it must hold it. The height matters: sitting back on the rump with the
  front legs up has the same pitch at about 0.2 m. Staying upright or low speed alone is not the pose.
- Every request that names something measurable needs a clause that measures it: a pose
  needs a pose metric, walking forward or backward needs forward_velocity, spinning needs
  yaw_rate. Staying upright and not falling never count as the task on their own.
- When nothing above measures the task, write code: a function check(traj) over one episode
  (contract and keys below, with the scene), returning named bool or float scalars, and clauses
  on them as check:<name> (bools become the fraction of episodes where they held, so
  check:climbed >= 0.8; floats their mean). Use it for stairs climbed, a ramp walked up and stopped
  on, poles weaved through, a ball dodged, a pose held, anything in order. Prefer the named
  metrics and outcomes above when one fits exactly; never use code to say the same thing weaker.
- Every check needs at least one clause on the task itself. fell, scene_fell, target_fell and
  upright never count on their own: "stand on two legs" is not "didn't fall".
- A code check must fail for a robot that just stands where it starts (unless standing still
  is the task): it is tested on one, and rejected if it passes.
- Measure over traj["valid"] steps only (jnp.where(traj["valid"] > 0, x, fallback)); for "for N
  seconds", count consecutive steps (a cumulative sum that resets, e.g. via jax.lax.associative_scan
  or a jnp.cumsum trick) and compare with N / 0.02. For order ("A then B"), compare the first step
  index each holds (jnp.argmax on a bool array gives the first True).
- Code is pure jax.numpy, run under jax.jit and vmap: no Python if/while on traced values (use
  jnp.where), no imports beyond jax, jax.numpy and math, no I/O.
- Only if the request is impossible to measure even with code (a mood, a style with no physical
  sign), return unsupported_reason.
"""

SCHEMA = {
    "type": "object",
    "properties": {
        "unsupported_reason": {"type": ["string", "null"]},
        "code": {"type": ["string", "null"]},  # check(traj) source, when a clause names check:<key>
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
    "required": ["clauses", "unsupported_reason", "code"],
    "additionalProperties": False,
}


class SpecError(ValueError):
    pass


def metric_definition(metric: str, scene: SceneSpec | None = None, check_keys: dict[str, str] | None = None):
    if metric in METRICS:
        return METRICS[metric]
    if metric.startswith(CHECK_PREFIX):
        key = metric[len(CHECK_PREFIX):]
        kind = (check_keys or {}).get(key)
        if kind is None:
            raise SpecError(f"{metric} names no result of the check code"
                            + (f"; it returns {', '.join(sorted(check_keys))}" if check_keys else "; write check(traj)"))
        if kind == "bool":
            return (f"fraction of episodes where the check's {key} held", 0.0, 1.0)
        return (f"mean over episodes of the check's {key}", *CODE_RANGE)
    try:
        mode, names = outcomes.parts(metric, scene or EMPTY)
    except ValueError as exc:
        raise SpecError(str(exc)) from exc
    description = (" then " if mode == "sequence" else " and ").join(outcomes.event(name, scene or EMPTY).description for name in names)
    return ("fraction of episodes completing: " + description, 0.0, 1.0)


def validate(clauses: Any, scene: SceneSpec | None = None, check_keys: dict[str, str] | None = None) -> list[dict[str, Any]]:
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
            definition = metric_definition(metric, scene, check_keys)
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


POSE_REQUEST = re.compile(
    r"\b(?:(?:two|hind|back|rear|front|three)\s+(?:legs?|paws?|feet))\b|\brear(?:s|ing|ed)?\s+up\b"
    r"|\b(?:lift|raise|hold\s+up)\w*\s+(?:\w+\s+){0,3}(?:paws?|legs?|feet|foot)\b|\bhandstands?\b"
    r"|\bsit(?:s|ting)?\s+up\b|\bbeg(?:s|ging)?\b|\bwheelbarrow\b",
    re.I,
)
# What a request names -> the metrics, any one of which measures it.
CORE_METRICS = {
    "pose": frozenset(POSE_METRICS),
    "forward": frozenset({"forward_velocity", "abs_forward_velocity"}),
    "backward": frozenset({"forward_velocity"}),
    "spin": frozenset({"yaw_rate", "abs_yaw_rate"}),
}


def core_intents(request: str) -> set[str]:
    """What a request names that a rollout metric can measure (pose, forward, backward, spin)."""
    found = {name for name in ("forward", "backward", "spin") if re.search(ACTION_WORDS[name], request, re.I)}
    if POSE_REQUEST.search(request):
        found.add("pose")
    return found


def requires_reach(request: str) -> bool:
    return bool(REACH_REQUEST.search(request))


def required_actions(request: str) -> set[str]:
    actions = {name for name, pattern in ACTION_WORDS.items() if re.search(pattern, request, re.I)}
    if requires_reach(request) and not actions & {"push", "climb"}:
        actions.add("reach")
    return actions


ANY_KIND = frozenset({"reach", "push", "climb", "avoid", "stand_still", "forward", "backward", "spin", "pose"})


def _kinds(metric: str, scene: SceneSpec) -> set[str]:
    if metric.startswith(CHECK_PREFIX):
        return set(ANY_KIND)  # the code measures whatever it was written for; the critic reads it
    if metric == "target_reached":
        return {"reach"}
    if metric in METRICS:
        return set()
    _, names = outcomes.parts(metric, scene)
    kinds = {name.split("@")[0].split(":")[0] for name in names}
    if "stand_on" in kinds:
        kinds.update(("stand_still", "climb"))
    return kinds


EPISODE_FALLS = frozenset({"scene_fell", "target_fell"})


def default_scene_check(scene: SceneSpec | None) -> list[dict[str, Any]] | None:
    """The check a scene's layout implies, for when Claude's can't be validated: deliver the
    target to its goal zone, or reach a fixed target, without falling. None for other scenes."""
    scene = scene or EMPTY
    target, goal = scene.target_object, scene.goal_object
    if target is not None and goal is not None and goal.kind == "zone":
        return [{"metric": f"push:{target.name}:{goal.name}", "op": ">=", "value": 0.8},
                {"metric": "scene_fell", "op": "<=", "value": 0.1}]
    if target is not None and not target.movable:
        return [{"metric": "target_reached", "op": ">=", "value": 0.8},
                {"metric": "target_fell", "op": "<=", "value": 0.1}]
    return None


def validate_for_scene(
    clauses: Any, request: str, scene: SceneSpec | None, check: "Check | None" = None,
) -> list[dict[str, Any]]:
    """Require concrete, scene-valid outcomes and same-episode/order constraints.
    With a fall bound over the evaluation episodes, a `fell` clause on the single video
    rollout is dropped: two fall criteria disagree on the board (37% vs 0).
    check carries the gated check code's result names and standing-robot values."""
    scene = scene or EMPTY
    spec = validate(clauses, scene, check.keys if check else None)
    if any(c["metric"] in EPISODE_FALLS for c in spec):
        spec = [c for c in spec if c["metric"] != "fell"]
    target = scene.target_object
    if any(c["metric"] in TARGET_METRICS for c in spec) and (target is None or target.movable):
        raise SpecError("target metrics require a fixed scene target; use named outcomes for movable objects")
    actions = required_actions(request)
    goal = scene.goal_object
    if goal is not None and goal.kind == "zone" and "reach" in actions:
        # "move the ball to the destination" reads as reaching, but the scene says the ball must
        # get to its goal zone: that's a push, and target_reached can't measure a movable target.
        actions = (actions - {"reach"}) | {"push"}
    ordered = bool(re.search(r"\b(?:then|after|before)\b", request, re.I))
    if not ordered and actions <= {"forward", "backward", "spin"}:
        actions = set()
    positive = [c for c in spec if c["op"] in (">", ">=") and c["value"] > 0]
    if actions or ordered:
        candidates = [c for c in positive if not ordered or c["metric"].startswith("sequence:")]
        if not any(actions <= _kinds(c["metric"], scene) for c in candidates):
            raise SpecError("The check needs a positive lower bound covering " + ", ".join(sorted(actions)) +
                            (" in a sequence" if ordered else " together in each episode") +
                            ("; moving the target to its goal is push:" + f"{scene.target}:{scene.goal}"
                             if "push" in actions and goal is not None else
                             "; reaching requires reach:OBJECT" + ("" if scene.target_object is not None
                                                                     and scene.target_object.movable
                                                                     else " or target_reached")))
        if not any(c["metric"] in ("target_fell", "scene_fell") and c["op"] in ("<", "<=") and c["value"] < 1 for c in spec):
            raise SpecError("Scene tasks require an upper bound on target_fell or scene_fell across the same episodes")
    for intent in sorted(core_intents(request)):
        # A named outcome such as sequence:forward>stand_still measures it too.
        if not any(c["metric"] in CORE_METRICS[intent] or intent in _kinds(c["metric"], scene) for c in spec):
            raise SpecError(f"The request asks for {intent} motion, so the check needs a clause on one of "
                            + ", ".join(sorted(CORE_METRICS[intent]))
                            + "; staying upright, a height or low speed doesn't show it")
    durations = [float(value) for value in re.findall(r"(?:for|still)\s+(\d+(?:\.\d+)?)\s*(?:seconds?\b|s\b)", request, re.I)]
    if durations:
        held = []
        for clause in positive:
            if clause["metric"].startswith(CHECK_PREFIX):
                held.append(math.inf)  # code measures durations itself; the critic reads the code
            elif clause["metric"] not in METRICS:
                _, names = outcomes.parts(clause["metric"], scene)
                held.extend(outcomes.event(name, scene).hold_s for name in names)
        remaining = sorted(held)
        for duration in sorted(durations, reverse=True):
            if not remaining or remaining[-1] < duration:
                raise SpecError("The check must measure every requested holding duration")
            remaining.pop()
    if not any(c["metric"] not in GENERIC_METRICS for c in spec):
        raise SpecError("The check needs at least one clause on the task itself; fell, scene_fell, target_fell and "
                        "upright only say it didn't fall. Use a metric or outcome that measures the behavior, "
                        "or write check(traj) code that does")
    if check is not None and check.code:
        _reject_trivial(spec, request, check)
    return spec


def _reject_trivial(spec: list[dict[str, Any]], request: str, check: Check) -> None:
    """A code check whose task clauses all hold for a robot standing still where it starts
    proves nothing, unless standing still is what was asked."""
    if re.search(ACTION_WORDS["stand_still"], request, re.I) or re.search(r"\b(?:stay|hold) (?:still|put)\b", request, re.I):
        return
    task = [c for c in spec if c["metric"] not in GENERIC_METRICS]
    if not task or not all(c["metric"].startswith(CHECK_PREFIX) for c in task):
        return  # a vocabulary metric carries the task
    holds = all(OPS[c["op"]](check.standing[c["metric"][len(CHECK_PREFIX):]], c["value"]) for c in task)
    if holds:
        values = ", ".join(f"{k} = {v:g}" for k, v in check.standing.items())
        raise SpecError("The check passes for a robot that just stands where it starts (its results there: "
                        f"{values}), so it doesn't show the task was done. Make the clauses fail unless the "
                        "behavior happens")


def gate_code(code: str | None, scene: SceneSpec | None) -> Check:
    """Run the check code through the gate (sim/check_code.py); raise SpecError for the checker."""
    if not code:
        return Check(clauses=[])
    from ..sim import check_code

    try:
        report = check_code.gate(code, scene or EMPTY)
    except check_code.CheckCodeError as exc:
        raise SpecError(f"The check code was rejected: {exc}") from exc
    return Check(clauses=[], code=code, keys=report.keys, standing=report.standing)


def validate_check(clauses: Any, code: str | None, request: str, scene: SceneSpec | None) -> Check:
    """Gate the code, then validate the clauses against its results."""
    check = gate_code(code, scene)
    uses_code = isinstance(clauses, list) and any(
        isinstance(c, dict) and str(c.get("metric", "")).startswith(CHECK_PREFIX) for c in clauses)
    if code and not uses_code:
        raise SpecError("The check code is never used: name its results in clauses as check:<name>, or drop the code")
    check.clauses = validate_for_scene(clauses, request, scene, check)
    return check


class ClaudeChecker:
    def __init__(self, client: anthropic.Anthropic, model: str):
        self.client, self.model = client, model

    def write_spec(self, request: str, scene: SceneSpec | None = None) -> list[dict[str, Any]]:
        return self.write_check(request, scene).clauses

    def write_check(self, request: str, scene: SceneSpec | None = None) -> Check:
        from ..sim import check_code

        context = scene.summary() if scene else "No scene target: the robot is alone on a flat floor."
        events = "\n".join(f"- {name}: {event.description}" for name, event in outcomes.catalog(scene or EMPTY).items())
        prompt = (f"Request: {request}\n\nScene:\n{context}\n\nAvailable episode outcomes:\n{events}\n\n"
                  f"For check(traj) code, {check_code.contract(scene or EMPTY)}\n\n"
                  "Write the success check. Use unsupported_reason=null for supported tasks, and code=null "
                  "when no clause names check:<key>.")
        for attempt in range(CHECK_ATTEMPTS):
            out = ask_json(self.client, self.model, SYSTEM, [{"type": "text", "text": prompt}], SCHEMA, effort="medium")
            if out.get("unsupported_reason"):
                raise SpecError(f"Unsupported task: {out['unsupported_reason']}")
            try:
                return validate_check(out.get("clauses"), out.get("code"), request, scene)
            except SpecError as exc:
                if attempt == CHECK_ATTEMPTS - 1:
                    raise
                prompt += (f"\n\nYour last check was rejected: {exc}\nIt was: {json.dumps(out.get('clauses'))}"
                           + (f"\nwith code:\n{out['code']}" if out.get("code") else ""))
        raise AssertionError("unreachable")


CHECK_ATTEMPTS = 3
