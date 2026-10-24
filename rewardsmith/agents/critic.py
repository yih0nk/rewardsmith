import base64
import json

import anthropic

from ..sim.base import RunResult
from .base import Review
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

Hard limits (they are also enforced in code):
- If the robot fell (metric "fell" is 1), the score is at most 3.
- If you flag reward hacking, the score is at most 3.
- If the measured motion contradicts the request (wrong direction, or near zero speed for a
  moving behavior), the score is at most 2.

Metrics: forward_velocity and lateral_velocity are mean m/s in the robot's own frame
(negative forward means backwards, positive lateral means left); yaw_rate is mean rad/s;
base_height is mean metres (about 0.28 standing); upright is the mean up-axis z (1 = level);
fell is 1 if it flipped; term_* are per-term episode rewards from the reward function.

Reward hacking means the robot collects reward without doing the behavior, for example
vibrating in place, sliding, or exploiting one term. Check the frames against the numbers.

Start the critique with the 2-3 numbers your score rests on. Then say exactly what the next
reward writer should change."""

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

    def review(self, request: str, result: RunResult) -> Review:
        content: list[dict] = []
        for png in result.frames[:MAX_FRAMES]:
            content.append({
                "type": "image",
                "source": {"type": "base64", "media_type": "image/png", "data": base64.standard_b64encode(png).decode()},
            })
        curve = ", ".join(f"{x:.2f}" for x in result.reward_curve[-20:])
        content.append({
            "type": "text",
            "text": (
                f"Requested behavior: {request}\n\n"
                f"Frames above are in time order.\n"
                f"Metrics: {json.dumps(result.metrics)}\n"
                f"Episode reward over training (last 20 evals): {curve}"
            ),
        })
        out = ask_json(self.client, self.model, SYSTEM, content, SCHEMA)
        return apply_limits(float(out["score"]), bool(out["reward_hacking"]), out["critique"], result.metrics)


FELL_OR_HACKED_MAX = 3.0


def apply_limits(score: float, reward_hacking: bool, critique: str, metrics: dict) -> Review:
    """Clamp to 0-10 and enforce the hard limits from the prompt, whatever Claude returned."""
    score = max(0.0, min(10.0, score))
    notes = []
    if float(metrics.get("fell", 0) or 0) >= 1 and score > FELL_OR_HACKED_MAX:
        score = FELL_OR_HACKED_MAX
        notes.append("Capped at 3: the robot fell.")
    if reward_hacking:
        notes.append("Reward hacking suspected.")
        score = min(score, FELL_OR_HACKED_MAX)
    return Review(score=score, critique=" ".join(notes + [critique]))
