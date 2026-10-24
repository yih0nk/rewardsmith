"""Reward revision between rounds.

DroidEditor is the target (Factory prize): triggered by a finished round, its
workspace holds only this request's reward folder, and every change is a diff.
ClaudeEditor is the fallback so the loop works before Droid is wired in.
"""

import difflib
import json
from typing import Any, Sequence

import anthropic

from .base import Attempt, Draft
from .claude import ask_json
from .writer import SYSTEM, WRITE_SCHEMA


def unified_diff(old: str, new: str) -> str:
    return "".join(
        difflib.unified_diff(
            old.splitlines(keepends=True), new.splitlines(keepends=True), "reward.py (before)", "reward.py (after)"
        )
    )


MAX_HISTORY = 12
MAX_CRITIQUE = 500


def format_history(history: Sequence[Attempt]) -> str:
    lines = []
    for a in list(history)[-MAX_HISTORY:]:
        critique = a.critique if len(a.critique) <= MAX_CRITIQUE else a.critique[:MAX_CRITIQUE] + "..."
        lines.append(f"- Round {a.round}, candidate {a.idx}, score {a.score:.1f}: {critique}")
    return "\n".join(lines)


class ClaudeEditor:
    def __init__(self, client: anthropic.Anthropic, model: str):
        self.client, self.model = client, model

    def revise(
        self,
        request: str,
        env_doc: str,
        best_source: str,
        critique: str,
        metrics: dict[str, Any],
        k: int,
        history: Sequence[Attempt] = (),
    ) -> list[Draft]:
        earlier = (
            f"Earlier attempts and what the critic said (don't repeat what failed):\n{format_history(history)}\n\n"
            if history
            else ""
        )
        prompt = (
            f"Environment notes:\n{env_doc}\n\n"
            f"Behavior the user wants: {request}\n\n"
            f"Best reward so far:\n```python\n{best_source}\n```\n\n"
            f"Metrics from its rollout:\n{json.dumps(metrics, indent=2)}\n\n"
            f"Critic's notes after watching the video:\n{critique}\n\n"
            f"{earlier}"
            f"Write {k} revised reward modules that fix what the critic saw. "
            "Each idea should say what you changed and why."
        )
        out = ask_json(self.client, self.model, SYSTEM, [{"type": "text", "text": prompt}], WRITE_SCHEMA)
        return [Draft(c["source"], unified_diff(best_source, c["source"]), c["idea"]) for c in out["candidates"]][:k]


class DroidEditor:
    """TODO(B): wire the Factory Droid SDK here.

    Plan: on round_completed, copy rewards/<request_id>/reward.py plus critique.md
    and metrics.json into a scratch workspace, run Droid with edit access to that
    folder only, read back the edited reward.py, and return it with its diff.
    Check the SDK docs at the Factory booth for the exact API.
    """

    def revise(self, *args: Any, **kwargs: Any) -> list[tuple[str, str]]:
        raise NotImplementedError("Droid editor not wired yet; use ClaudeEditor")
