"""Ranks candidates mid-training for successive halving (rewardsmith.search).

No video yet: the ranker sees each robot's measured behavior at the same eval,
which compares across rewards, and the reward terms, which can show a policy
farming one term.
"""

import json
from typing import Any

import anthropic

from .claude import ask_json
from .robot import for_body

SYSTEM = """You decide which robot-learning runs keep training. Each run trains a simulated
Unitree Go1 quadruped with PPO on a different reward function for the same requested
behavior. You see every run's measurements at the same point early in training.

Rank the runs from most to least promising for the requested behavior. Judge the behavior,
not the reward: rewards differ between runs, so episode_reward and term_* values are not
comparable across runs. Use term_* only to spot a policy farming one term.

Measurements (means over 128 evaluation episodes): forward_velocity and lateral_velocity in
m/s in the robot's frame (negative forward is backwards, positive lateral is left); yaw_rate
in rad/s; base_height in m (about 0.28 standing); upright is the up-axis z (1 = level);
episode_length in steps out of 1000 (shorter means robots fell); steps is training steps so far.

A run that already moves the right way, stays upright and does not fall is more promising
than one that is still standing still. Prefer steady progress toward the request over a
higher number that comes from falling or flailing."""

SCHEMA = {
    "type": "object",
    "properties": {
        "order": {"type": "array", "items": {"type": "integer"}},
        "reason": {"type": "string"},
    },
    "required": ["order", "reason"],
    "additionalProperties": False,
}


class ClaudeRanker:
    body: str | None = None  # the request\'s robot; the worker sets it per request

    def __init__(self, client: anthropic.Anthropic, model: str):
        self.client, self.model = client, model

    def rank(self, request: str, snapshots: list[dict[str, Any]]) -> list[int]:
        runs = "\n".join(f"run {i}: {json.dumps(_rounded(s))}" for i, s in enumerate(snapshots))
        prompt = (
            f"Requested behavior: {request}\n\n{runs}\n\n"
            f"Return order: all run numbers 0-{len(snapshots) - 1}, most promising first."
        )
        out = ask_json(self.client, self.model, for_body(SYSTEM, self.body), [{"type": "text", "text": prompt}], SCHEMA, effort="low")
        return [int(i) for i in out["order"]]


def _rounded(snapshot: dict[str, Any]) -> dict[str, Any]:
    return {k: round(v, 3) if isinstance(v, float) else v for k, v in snapshot.items()}
