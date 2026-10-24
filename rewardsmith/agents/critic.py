import base64
import json

import anthropic

from ..sim.base import RunResult
from .base import Review
from .claude import ask_json

SYSTEM = """You judge robot learning results. You see frames from a simulated rollout,
the training curve and measured metrics. Score how well the behavior matches the
user's request from 0 (nothing like it) to 10 (clearly does it, stably).
Watch for reward hacking: the robot satisfying the numbers without doing the behavior.
Your critique tells the next reward writer exactly what to change."""

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
