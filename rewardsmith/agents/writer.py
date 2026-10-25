import anthropic

from .base import REWARD_CONTRACT, Draft
from .claude import ask_json
from .stream import IdeaStream, OnIdea

SYSTEM = f"""You design reward functions for reinforcement learning in MuJoCo Playground (JAX).
A PPO policy will be trained on your reward, so it must be dense, smooth and hard to exploit.

Every reward you write is a complete Python module with this contract:

{REWARD_CONTRACT}
Rules:
- Use only jax and jax.numpy. No Python control flow on traced values; use jnp.where.
- Return a dict of named terms (e.g. "backward_velocity", "upright", "energy").
  Named terms let us see which term the policy is exploiting.
- Keep scales so no single term dominates unless it should.
"""

WRITE_SCHEMA = {
    "type": "object",
    "properties": {
        "candidates": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"idea": {"type": "string"}, "source": {"type": "string"}},
                "required": ["idea", "source"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["candidates"],
    "additionalProperties": False,
}

FIX_SCHEMA = {
    "type": "object",
    "properties": {"source": {"type": "string"}},
    "required": ["source"],
    "additionalProperties": False,
}


class ClaudeWriter:
    def __init__(self, client: anthropic.Anthropic, model: str):
        self.client, self.model = client, model

    def write(self, request: str, env_doc: str, k: int, on_idea: OnIdea | None = None) -> list[Draft]:
        """on_idea(i, idea) is called as each candidate's idea streams in."""
        prompt = (
            f"Environment notes:\n{env_doc}\n\n"
            f"Behavior the user wants: {request}\n\n"
            f"Write {k} different reward modules. Make them genuinely different approaches, "
            "not small weight changes of one idea."
        )
        out = ask_json(
            self.client, self.model, SYSTEM, [{"type": "text", "text": prompt}], WRITE_SCHEMA,
            on_text=IdeaStream(on_idea) if on_idea else None,
        )
        return [Draft(c["source"], idea=c["idea"]) for c in out["candidates"]][:k]

    def fix(self, source: str, error: str, env_doc: str) -> str:
        prompt = (
            f"Environment notes:\n{env_doc}\n\n"
            f"This reward module failed to compile or run:\n```python\n{source}\n```\n\n"
            f"Error:\n{error}\n\nReturn the corrected module. Change only what the error requires."
        )
        out = ask_json(
            self.client, self.model, SYSTEM, [{"type": "text", "text": prompt}], FIX_SCHEMA, effort="medium"
        )
        return out["source"]
