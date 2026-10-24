from dataclasses import dataclass
from typing import Any, Protocol

from ..sim.base import RunResult
from ..sim.scene import SceneSpec

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
    def review(self, request: str, result: RunResult, check: dict[str, Any] | None = None) -> Review:
        """check is checker.evaluate()'s result for this run, when the request has a success spec."""
        ...


class Checker(Protocol):
    def write_spec(self, request: str, scene: SceneSpec | None = None) -> list[dict[str, Any]]:
        """A validated success spec (see checker.py), written before any training."""
        ...


class Ranker(Protocol):
    def rank(self, request: str, snapshots: list[dict[str, Any]]) -> list[int]:
        """Positions into snapshots, most promising first (successive halving)."""
        ...


class Scener(Protocol):
    """Turns a request into a scene spec dict (see sim/scene.py), before any reward is written."""

    def make(self, request: str) -> dict[str, Any]: ...

    def fix(self, request: str, spec: dict[str, Any], error: str) -> dict[str, Any]: ...
