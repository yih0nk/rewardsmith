from dataclasses import dataclass
from typing import Any, Protocol

from ..sim.base import RunResult

# Every reward a writer or editor produces must define exactly this function.
REWARD_CONTRACT = '''\
import jax
import jax.numpy as jnp

def reward(state, action, prev_action) -> dict[str, jax.Array]:
    """Return named reward terms as scalars. The trainer sums them.
    Use only jax / jnp and the fields described in the environment notes."""
'''


@dataclass
class Review:
    score: float  # 0 to 10: how well the rollout matches the request
    critique: str  # what went wrong and what to change


class Writer(Protocol):
    def write(self, request: str, env_doc: str, k: int) -> list[str]: ...

    def fix(self, source: str, error: str, env_doc: str) -> str: ...


class Editor(Protocol):
    def revise(
        self,
        request: str,
        env_doc: str,
        best_source: str,
        critique: str,
        metrics: dict[str, Any],
        k: int,
    ) -> list[tuple[str, str]]:
        """Return k (new_source, unified_diff) pairs."""
        ...


class Critic(Protocol):
    def review(self, request: str, result: RunResult) -> Review: ...


class Scener(Protocol):
    """Turns a request into a scene spec dict (see sim/scene.py), before any reward is written."""

    def make(self, request: str) -> dict[str, Any]: ...

    def fix(self, request: str, spec: dict[str, Any], error: str) -> dict[str, Any]: ...
