"""Offline stand-ins so the loop and UI run with no API key."""

from typing import Any

from ..sim.base import RunResult
from .base import Review
from .editor import unified_diff

TEMPLATE = '''\
import jax
import jax.numpy as jnp

def reward(state, action, prev_action):
    # fake draft {n}
    return {{
        "task": jnp.asarray({w}),
        "energy": -0.01 * jnp.sum(jnp.square(action)),
    }}
'''


class FakeWriter:
    def write(self, request: str, env_doc: str, k: int) -> list[str]:
        return [TEMPLATE.format(n=i, w=1.0 + i * 0.25) for i in range(k)]

    def fix(self, source: str, error: str, env_doc: str) -> str:
        return source


class FakeEditor:
    def revise(
        self, request: str, env_doc: str, best_source: str, critique: str, metrics: dict[str, Any], k: int
    ) -> list[tuple[str, str]]:
        out = []
        for i in range(k):
            new = best_source.rstrip("\n") + f"\n# fake revision {i}\n"
            out.append((new, unified_diff(best_source, new)))
        return out


class FakeCritic:
    def review(self, request: str, result: RunResult) -> Review:
        score = float(result.metrics.get("fake_quality", 5.0))
        return Review(score=score, critique=f"Fake critique for '{request}': try a stronger task term.")
