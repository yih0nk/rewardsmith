import base64
import json
from typing import Any

import anthropic

from ..sim.base import RunResult
from ..sim.go1_spec import POSE_METRICS
from .base import Review
from .checker import METRICS, describe
from .claude import ask_json

SYSTEM = """You are a strict judge of robot learning results. You see frames from a
simulated rollout of a Unitree Go1 quadruped (in time order), measured metrics, and the
training reward curve. Score how well the robot does the user's requested behavior.

Be calibrated, not generous. Scores must spread out so better attempts are visible:
- 0-2: does not do it, does the opposite, or barely moves.
- 3-4: a weak or inconsistent version (slow, stumbling, mostly something else).
- 5-6: clearly attempts it but sloppy (drift, wobble, odd gait, too slow).
- 7: does it reliably, with one obvious flaw.
- 8: does it cleanly and stably at a good pace; only minor flaws.
- 9: excellent: fast or precise, natural gait, no drift, no wasted motion.
- 10: essentially perfect. Almost never give 10.
A first attempt that merely works is a 6 or 7. Give 8 or more only if you can name what
makes it better than merely working.

Hard limits (the first three are also enforced in code):
- If the robot fell (metric "fell" is 1), the score is at most 3.
- If you flag reward hacking, the score is at most 3.
- If the success check's task clause failed (the task didn't get done), the score is at most 4.
- If the measured motion contradicts the request (wrong direction, or near zero speed for a
  moving behavior), the score is at most 2.

Metrics: forward_velocity and lateral_velocity are mean m/s in the robot's own frame
(negative forward means backwards, positive lateral means left); yaw_rate is mean rad/s;
base_height is mean metres (about 0.28 standing); upright is the mean up-axis z (1 = level);
fell is 1 if it flipped; term_* are per-term episode rewards from the reward function.
Pose metrics, over the same rollout: front_feet_off and rear_feet_off are the fraction of time
both front (or both rear) feet are off the floor, front_off_held_s and rear_off_held_s the longest
unbroken stretch of that in seconds, feet_off_mean the mean number of feet in the air, pitch_deg
the mean body pitch (nose up positive; about 60-90 standing on the hind legs) and abs_roll_deg
the mean roll. For a pose request, judge the pose from these and the frames, not from height.
When the scene has a target object, the task metrics come from 32 fresh 10 s episodes. The
target_* ones are about the ROBOT'S BASE and the target: target_reached is the fraction that got
within reach of it, target_final_in the fraction that ended within reach, target_start_dist,
target_min_dist and target_final_dist mean distances in metres, target_time_to_reach mean seconds to
reach it (-1 if never), target_fell the fraction that flipped, final_speed the base speed over the
last second. For moving an object, object_moved is how far the object moved (m), and the
object_goal_* ones are about the OBJECT and the goal: object_goal_start/min/final_dist (m),
object_goal_reached (fraction ever in the goal), object_in_goal_final (fraction ending in it) and
object_goal_progress (fraction of the distance closed; negative = pushed away). The robot reaching
the ball is not the ball reaching the goal. phase_* give the fraction of episodes stuck at each
step of the task. Named outcomes such as push:ball:goal, reach:box or climb:step are the
fraction of the same episodes that completed that outcome, and scene_fell the fraction that
fell. Trust these over a single video for whether the task got done.

Reward hacking means the robot collects reward without doing the behavior, for example
vibrating in place, sliding, or exploiting one term. Check the frames against the numbers.

You may also get the success check: pass/fail clauses over these metrics, written from the
request before any training. The request only counts as done if the check passes AND your
score clears the bar. If a clause failed, say which one and what the reward should change
to fix it. Judge the motion yourself too: the check can pass while the behavior looks wrong.

Start the critique with the 2-3 numbers your score rests on. For a scene task, then name the
phase where it fails, using the task diagnosis you are given and the frames: never approached
the object; approached but never touched or moved it; moved it the wrong way (not behind it on
the far side from the goal); pushed toward the goal but stopped short; got it there then overshot
or pushed it out; or reached the target but didn't stop or stay. Then say exactly what the next
reward writer should change to get past that phase."""

SCHEMA = {
    "type": "object",
    "properties": {
        "score": {"type": "number"},
        "reward_hacking": {"type": "boolean"},
        "critique": {"type": "string"},
    },
    "required": ["score", "reward_hacking", "critique"],
    "additionalProperties": False,
}

MAX_FRAMES = 10


class ClaudeCritic:
    def __init__(self, client: anthropic.Anthropic, model: str):
        self.client, self.model = client, model

    def review(self, request: str, result: RunResult, check: dict[str, Any] | None = None) -> Review:
        content: list[dict] = []
        for png in result.frames[:MAX_FRAMES]:
            content.append({
                "type": "image",
                "source": {"type": "base64", "media_type": "image/png", "data": base64.standard_b64encode(png).decode()},
            })
        curve = ", ".join(f"{x:.2f}" for x in result.reward_curve[-20:])
        diagnosis = task_diagnosis(result.metrics)
        content.append({
            "type": "text",
            "text": (
                f"Requested behavior: {request}\n\n"
                f"Frames above are in time order.\n"
                f"Metrics: {json.dumps(result.metrics)}\n"
                f"Episode reward over training (last 20 evals): {curve}"
                + (f"\n\nSuccess check (written before training):\n{describe(check)}" if check else "")
                + (f"\n\nTask diagnosis (computed from the 32 episodes):\n{diagnosis}" if diagnosis else "")
            ),
        })
        out = ask_json(self.client, self.model, SYSTEM, content, SCHEMA)
        return apply_limits(float(out["score"]), bool(out["reward_hacking"]), out["critique"], result.metrics, check)


FELL_OR_HACKED_MAX = 3.0
TASK_FAILED_MAX = 4.0  # the task-completion clause of the success check failed
# Checker metrics whose clause says a scene task got done, beside the named outcomes
# (push:ball:goal, reach:box, ...), which are never in checker.METRICS.
TARGET_TASK_METRICS = frozenset({"target_reached", "target_final_dist"}) | frozenset(POSE_METRICS)


def is_task_metric(metric: str) -> bool:
    """True for a success-check clause about the task itself (a scene outcome, reaching the
    target, or a pose), not about falling or speed."""
    return metric in TARGET_TASK_METRICS or metric not in METRICS

# What each phase means, and what the reward should change to get past it (scene_eval phases).
PHASES = {
    "no_approach": ("never got to the object",
                    "make approaching it pay: speed toward the object (or toward the spot behind it, for a push)"),
    "no_push": ("got to the object but barely moved it (under 15 cm)",
                "pay for the object closing on the goal (target_to_goal_speed), and cut terms that pay for "
                "standing near it or facing it, which it is farming"),
    "wrong_way": ("moved the object, but not toward the goal",
                  "get behind it first, on the far side from the goal (robot-to-object direction lined up with "
                  "object-to-goal), and pay only for progress toward the goal"),
    "short": ("pushed it toward the goal but stopped short",
              "raise the progress term against the posture and proximity terms, and keep pushing until it's in"),
    "overshoot": ("got it into the goal, then pushed it out again",
                  "add a bonus for the object being in the goal, and stop paying for pushing once it's there"),
    "left": ("reached the target but did not end there",
             "add a bounded arrival bonus and a stop (low speed) term once within reach"),
}
SUCCESS_PHASES = ("delivered", "arrived")


def task_diagnosis(metrics: dict[str, Any]) -> str | None:
    """Which phase of the scene task the episodes get stuck at, from scene_eval's phase_* metrics."""
    phases = {k.removeprefix("phase_"): float(v) for k, v in metrics.items()
              if k.startswith("phase_") and isinstance(v, (int, float))}
    if not phases:
        return None
    shares = ", ".join(f"{name} {frac:.0%}" for name, frac in phases.items())
    done = sum(phases.get(p, 0.0) for p in SUCCESS_PHASES)
    stuck = max((p for p in phases if p in PHASES), key=lambda p: phases[p], default=None)
    if stuck is None or phases[stuck] == 0:
        return f"Episodes by phase: {shares}. The task got done in {done:.0%} of episodes."
    what, fix = PHASES[stuck]
    return (f"Episodes by phase: {shares}. Done in {done:.0%}. Most episodes fail at '{stuck}' "
            f"({phases[stuck]:.0%}): it {what}. To get past it: {fix}.")


def apply_limits(
    score: float, reward_hacking: bool, critique: str, metrics: dict, check: dict[str, Any] | None = None
) -> Review:
    """Clamp to 0-10 and enforce the hard limits from the prompt, whatever Claude returned.
    The critique leads with the caps and the task diagnosis, for the editor."""
    score = max(0.0, min(10.0, score))
    notes = []
    if float(metrics.get("fell", 0) or 0) >= 1 and score > FELL_OR_HACKED_MAX:
        score = FELL_OR_HACKED_MAX
        notes.append("Capped at 3: the robot fell.")
    if reward_hacking:
        notes.append("Reward hacking suspected.")
        score = min(score, FELL_OR_HACKED_MAX)
    failed = [c for c in (check or {}).get("clauses", []) if is_task_metric(c["metric"]) and not c["pass"]]
    if failed and score > TASK_FAILED_MAX:
        score = TASK_FAILED_MAX
        why = "; ".join(f"{c['metric']} {c['op']} {c['value']:g}, got {_num(c['actual'])}" for c in failed)
        notes.append(f"Capped at 4: the task was not done ({why}).")
    diagnosis = task_diagnosis(metrics)
    return Review(score=score, critique=" ".join(notes + ([diagnosis] if diagnosis else []) + [critique]),
                  reward_hacking=reward_hacking)


def _num(x: Any) -> str:
    return f"{x:.3g}" if isinstance(x, (int, float)) else "nothing"
