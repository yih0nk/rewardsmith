import anthropic

from .base import REWARD_CONTRACT, Draft
from .claude import ask_json
from .references import reference
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
- Keep the stock gait and stability terms in every reward: upright (up_z), a base height
  target near the standing height (stops crouching and belly-crawling), and small penalties on
  vertical bounce, roll and pitch rate, and action rate.
- Terms that pay every step whatever the robot does (upright, height, facing, being near
  something) must stay well below the task terms, or the policy parks and farms them.

Recipe for object tasks (the environment notes list the objects):
- Go to an object: pay for speed toward it while far (clipped to a brisk walk), a bounded
  arrival bonus within reach, and a stop term (penalize speed) once there.
- Move an object to a goal, in phases:
  1. approach: speed toward the push spot behind the object, on the far side from the goal
     (object position - (its radius + 0.3 m) * the unit object-to-goal direction), then
     toward the object itself once lined up;
  2. progress, the main term: state["target_to_goal_speed"], the per-step decrease of the
     object-to-goal distance (potential-based, so parking earns nothing), clipped to about
     [-0.5, 1] with a weight around 4;
  3. alignment: a small term for the robot-to-object direction matching the object-to-goal
     direction;
  4. success: a bonus while target_to_goal_dist is under the goal radius, with the approach
     and alignment terms switched off once the object is in.
  Success is measured on the object, never on the robot's distance to it.
- Vary the candidates in how they shape these phases (kernels, weights, the push spot), not
  by dropping a phase.
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
            f"{reference(request, env_doc)}"
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
