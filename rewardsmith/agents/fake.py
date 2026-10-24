"""Offline stand-ins so the loop and UI run with no API key."""

import re
from typing import Any

from ..sim import scene
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


TARGET_TEMPLATE = '''\
import jax
import jax.numpy as jnp

def reward(state, action, prev_action):
    # fake draft {n}: go to the target
    return {{
        "reach": {w} * jnp.exp(-state["target_dist"]),
        "face": 0.3 * jnp.cos(state["target_heading_err"]),
        "energy": -0.01 * jnp.sum(jnp.square(action)),
    }}
'''


class FakeWriter:
    def write(self, request: str, env_doc: str, k: int) -> list[str]:
        template = TARGET_TEMPLATE if 'state["target_dist"]' in env_doc else TEMPLATE
        return [template.format(n=i, w=1.0 + i * 0.25) for i in range(k)]

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


# Words that name a scene object, by kind. "target" is left out: it's often just "the target".
_KIND_WORDS = {
    "ball": ("ball", "sphere"),
    "box": ("box", "crate", "cube", "block", "step"),
    "zone": ("zone", "circle", "area", "spot", "square", "pad", "goal"),
}
_PUSH_WORDS = ("push", "kick", "dribble", "shove", "nudge", "roll")


class FakeScener:
    """Keyword scene writer: one object per kind named in the request, in its colour."""

    def make(self, request: str) -> dict[str, Any]:
        words = re.findall(r"[a-z]+", request.lower())
        movable = any(w.startswith(_PUSH_WORDS) for w in words)
        objects = []
        for kind, names in _KIND_WORDS.items():
            # A noun, not a verb ("circle it"): an article or a colour comes just before it.
            hits = [i for i, w in enumerate(words) if w.rstrip("s") in names
                    and set(words[max(0, i - 2):i]) & ({"the", "a", "an"} | set(scene.COLORS))]
            if not hits:
                continue
            before = words[max(0, hits[0] - 2):hits[0]]
            color = next((w for w in before if w in scene.COLORS), "red" if kind == "ball" else "blue")
            objects.append(_fake_object(kind, color, movable and kind != "zone", len(objects)))
        rough = any(w in ("rough", "bumpy", "uneven", "rocky") for w in words)
        return {
            "terrain": "rough" if rough else "flat",
            "objects": objects,
            "target": objects[0]["name"] if objects else None,
        }

    def fix(self, request: str, spec: dict[str, Any], error: str) -> dict[str, Any]:
        return self.make(request)


def _fake_object(kind: str, color: str, movable: bool, i: int) -> dict[str, Any]:
    size = {"ball": [0.15], "box": [0.4, 0.4, 0.3], "zone": [0.5]}[kind]
    return {
        "name": kind, "kind": kind, "color": color, "size": size,
        "distance": [1.5, 3.0], "bearing": [-180 + 120 * i, -60 + 120 * i], "yaw": [0, 0],
        "movable": movable, "mass": 0.5 if movable else 1.0,
    }


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
