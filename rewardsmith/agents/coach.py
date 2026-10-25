"""Watches one candidate train and says, at each eval, whether to keep it going.

The critic only sees a finished run. The coach sees the run while it trains: the
eval curve so far, each reward term, and what the robot is doing (behavior means),
so a candidate that farms one term or is falling over can be stopped minutes early
and its fix drafted while the others still train (see rewardsmith.coaching).

It runs on a fast model (Claude Haiku 5.5 by default) and never blocks training.
"""

import base64
import json
import time
from typing import Any, Sequence

import anthropic

from .base import Observation
from .checker import evaluate
from .claude import ask_json

DEFAULT_MODEL = "claude-haiku-5-5"
# $ per million tokens (input, output), for the board's cost line. Unknown models count as Sonnet.
PRICES = {"claude-haiku-5-5": (0.10, 0.50), "claude-sonnet-5-5": (2.0, 10.0), "claude-opus-5-5": (4.0, 20.0)}

SYSTEM = """You coach robot-learning runs while they train. A simulated Unitree Go1 quadruped
trains with PPO on one candidate reward function for the user's requested behavior. You see
that run at every evaluation so far and decide what happens next:

- continue: it is learning the requested behavior, or it is too early to tell.
- revise: it learns something, but the reward steers it wrong (one term dwarfs the task term,
  it is learning the wrong direction, it stands still to collect posture or alive bonuses).
  Training stops now and the reward is rewritten with your diagnosis.
- kill: hopeless; nothing about this reward is worth keeping (diverged, falls in every
  episode, reward terms that cannot produce the behavior).

Stopping a run that would have worked costs the user a good policy, so say continue unless
the numbers show a problem. Early evals are noisy: a run can stand still for a while before
it starts moving.

How to read the numbers (means over the evaluation episodes):
- terms: per-term episode reward. Compare their sizes: a term 10x or more larger than the
  term that rewards the task usually means the policy optimizes that term instead.
- behavior: forward_velocity and lateral_velocity in m/s in the robot's frame (negative
  forward is backwards, positive lateral is left), yaw_rate in rad/s, base_height in m (about
  0.28 standing), upright is the up-axis z (1 = level), episode_length in steps out of the
  episode's 1000 (shorter means robots fell and episodes ended early).
- Rewards differ between runs; judge the behavior, not the size of episode_reward.

The note goes on a live board the user watches: one plain-English sentence about what the
robot is doing and why, with the number that shows it. No term-name jargon without saying
what it means. The diagnosis is for the engineer who rewrites the reward: name the terms to
change, by how much, and why."""

SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["continue", "revise", "kill"]},
        "note": {"type": "string"},
        "diagnosis": {"type": "string"},
    },
    "required": ["verdict", "note", "diagnosis"],
    "additionalProperties": False,
}

MAX_FRAMES = 3


class ClaudeCoach:
    def __init__(self, client: anthropic.Anthropic, model: str = DEFAULT_MODEL):
        self.client, self.model = client, model

    def observe(
        self,
        request: str,
        source: str,
        curve: Sequence[dict[str, Any]],
        total_steps: int,
        spec: list[dict[str, Any]] | None = None,
        frames: Sequence[bytes] = (),
    ) -> Observation:
        content: list[dict[str, Any]] = [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                         "data": base64.standard_b64encode(png).decode()}}
            for png in frames[:MAX_FRAMES]
        ]
        content.append({"type": "text", "text": prompt(request, source, curve, total_steps, spec)})
        usage: list[Any] = []
        start = time.monotonic()
        # Haiku takes no server-side refusal fallback; low effort keeps the call to a few seconds.
        out = ask_json(
            self.client, self.model, SYSTEM, content, SCHEMA, effort="low", max_tokens=4000,
            fallback=not self.model.startswith("claude-haiku"), on_usage=usage.append,
        )
        latency = time.monotonic() - start
        tokens_in = getattr(usage[0], "input_tokens", 0) if usage else 0
        tokens_out = getattr(usage[0], "output_tokens", 0) if usage else 0
        return Observation(
            verdict=out["verdict"], note=out["note"].strip(), diagnosis=out["diagnosis"].strip(),
            latency_s=latency, cost_usd=cost(self.model, tokens_in, tokens_out),
        )


def prompt(
    request: str, source: str, curve: Sequence[dict[str, Any]], total_steps: int,
    spec: list[dict[str, Any]] | None = None,
) -> str:
    """The text the coach sees: request, check, reward, and one line per eval so far."""
    rows = "\n".join(f"eval {i}: {json.dumps(_rounded(p))}" for i, p in enumerate(curve))
    parts = [
        f"Requested behavior: {request}",
        f"Reward function being trained:\n```python\n{source}\n```",
        f"Training progress: {curve[-1].get('steps', 0) if curve else 0:,} of {total_steps:,} steps. "
        "Eval 0 is the untrained policy.",
        f"Evals so far:\n{rows}",
    ]
    if spec:
        clauses = "\n".join(f"- {c['metric']} {c['op']} {c['value']}" for c in spec)
        parts.append(f"Success check, measured on the finished policy:\n{clauses}")
        now = check_now(spec, curve[-1] if curve else {})
        if now:
            parts.append("Clauses the latest eval already measures:\n" + "\n".join(
                f"- {c['metric']} {c['op']} {c['value']}: now {c['actual']:.3g}, {'holds' if c['pass'] else 'fails'}"
                for c in now))
    parts.append("Decide: continue, revise or kill. Write the note and the diagnosis.")
    return "\n\n".join(parts)


def check_now(spec: list[dict[str, Any]], point: dict[str, Any]) -> list[dict[str, Any]]:
    """The success check's clauses an eval point can already answer (behavior means only)."""
    behavior = point.get("behavior") or {}
    measured = evaluate(spec, behavior)["clauses"]
    return [c for c in measured if c["actual"] is not None]


def cost(model: str, tokens_in: int, tokens_out: int) -> float:
    price_in, price_out = PRICES.get(model, PRICES["claude-sonnet-5-5"])
    return (tokens_in * price_in + tokens_out * price_out) / 1e6


def _rounded(point: dict[str, Any]) -> dict[str, Any]:
    def r(v: Any) -> Any:
        if isinstance(v, float):
            return round(v, 3)
        if isinstance(v, dict):
            return {k: r(x) for k, x in v.items()}
        return v

    return {k: r(v) for k, v in point.items() if k != "wall_s"}
