"""Offline stand-ins so the loop and UI run with no API key."""

from typing import Any

from ..sim.base import RunResult
from .base import Review
from .checker import NO_FALL, validate
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


class FakeChecker:
    """Keyword rules, so offline runs get a plausible spec."""

    def write_spec(self, request: str) -> list[dict[str, Any]]:
        text = request.lower()
        if "backward" in text:
            core = [{"metric": "forward_velocity", "op": "<", "value": -0.3}]
        elif "spin" in text or "turn" in text:
            core = [{"metric": "abs_yaw_rate", "op": ">", "value": 1.0},
                    {"metric": "abs_forward_velocity", "op": "<", "value": 0.3}]
        elif "walk" in text or "run" in text or "forward" in text:
            core = [{"metric": "forward_velocity", "op": ">", "value": 0.3}]
        else:
            core = [{"metric": "upright", "op": ">", "value": 0.8}]
        return validate(core + [NO_FALL])


class FakeCritic:
    def review(self, request: str, result: RunResult, check: dict[str, Any] | None = None) -> Review:
        score = float(result.metrics.get("fake_quality", 5.0))
        return Review(score=score, critique=f"Fake critique for '{request}': try a stronger task term.")


class FakeRanker:
    def rank(self, request: str, snapshots: list[dict[str, Any]]) -> list[int]:
        """Highest episode reward first: fine offline, though rewards don't compare in real runs."""
        return sorted(range(len(snapshots)), key=lambda i: -(snapshots[i].get("episode_reward") or 0.0))
