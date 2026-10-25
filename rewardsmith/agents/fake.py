"""Offline stand-ins so the loop and UI run with no API key."""

import re
from typing import Any

from ..sim import scene
from ..sim.base import RunResult
from .base import Review
from .checker import NO_FALL, SpecError, required_actions, requires_reach, validate_for_scene
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
    def write(self, request: str, env_doc: str, k: int, on_idea=None) -> list[str]:
        template = TARGET_TEMPLATE if 'state["target_dist"]' in env_doc else TEMPLATE
        for i in range(k if on_idea else 0):
            on_idea(i, f"fake draft {i}: task term weighted {1.0 + i * 0.25:g}")
        return [template.format(n=i, w=1.0 + i * 0.25) for i in range(k)]

    def fix(self, source: str, error: str, env_doc: str) -> str:
        return source


class FakeEditor:
    def revise(
        self, request: str, env_doc: str, best_source: str, critique: str, metrics: dict[str, Any], k: int,
        on_idea=None,
    ) -> list[tuple[str, str]]:
        out = []
        for i in range(k):
            if on_idea:
                on_idea(i, f"fake revision {i}")
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
    """Keyword scene writer: distinguish named colours and use low boxes for steps."""

    def make(self, request: str) -> dict[str, Any]:
        words = re.findall(r"[a-z]+", request.lower())
        objects = []
        for kind, names in _KIND_WORDS.items():
            hits = [i for i, w in enumerate(words) if w.rstrip("s") in names
                    and set(words[max(0, i - 2):i]) & ({"the", "a", "an"} | set(scene.COLORS))]
            colored = {}
            for hit in hits:
                before = words[max(0, hit - 2):hit]
                color = next((w for w in before if w in scene.COLORS), "red" if kind == "ball" else "blue")
                if color in colored:
                    continue  # mentioning the same coloured object again is not a new body
                prefix = " ".join(words[max(0, hit - 5):hit])
                movable = kind != "zone" and bool(re.search(r"\b(?:push\w*|kick\w*|dribbl\w*|shov\w*|nudg\w*)\b", prefix))
                value = _fake_object(kind, color, movable, len(objects) + len(colored))
                if kind == "box" and words[hit] == "step":
                    value["size"] = [0.6, 0.6, 0.1]
                colored[color] = value
            for color, value in colored.items():
                if len(colored) > 1:
                    value["name"] = f"{color}_{kind}"
                objects.append(value)
        objects = objects[:scene.MAX_OBJECTS]
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
        "distance": [1.5, 3.0], "bearing": [-180 + 60 * i, -120 + 60 * i], "yaw": [0, 0],
        "movable": movable, "mass": 0.5 if movable else 1.0,
    }


def _scene_plan(request: str, scene_spec: scene.SceneSpec | None):
    """Offline parser for the same named event vocabulary the real checker sees."""
    world = scene_spec or scene.EMPTY
    ordered = bool(re.search(r"\b(?:then|after|before)\b", request, re.I))
    actions = required_actions(request)
    if not ordered and actions <= {"reach", "forward", "backward", "spin"}:
        return None
    if not actions and not ordered:
        return None

    def choose(text, predicate):
        choices = [obj for obj in world.objects if predicate(obj)]
        if not choices:
            raise SpecError("The scene is missing an object required by this task")
        return max(choices, key=lambda obj: (3 * bool(re.search(r"\b" + re.escape(obj.name.replace("_", " ")) + r"\b", text))
                                             + 2 * bool(re.search(r"\b" + obj.color + r"\b", text))))

    segments = re.split(r"\b(?:then|after|before)\b", request.lower())
    if re.search(r"\bafter\b", request, re.I) and len(segments) == 2:
        segments.reverse()
    events = []
    previous_climb = None
    for text in segments:
        kinds = required_actions(text)
        stage = []
        if "push" in kinds:
            obj = choose(text, lambda obj: obj.movable)
            zone = choose(text, lambda obj: obj.kind == "zone")
            stage.append(f"push:{obj.name}:{zone.name}")
        if "climb" in kinds:
            box = choose(text, lambda obj: obj.kind == "box" and not obj.movable)
            previous_climb = box.name
            stage.append(f"climb:{box.name}")
        if "stand_still" in kinds:
            if "stand on" in text:
                previous_climb = choose(text, lambda obj: obj.kind == "box" and not obj.movable).name
            stage.append(f"stand_on:{previous_climb}" if previous_climb else "stand_still")
        obstacle = None
        if "avoid" in kinds:
            phrase = re.split(r"\b(?:avoid\w*|around|without touching|without hitting)\b", text)[-1]
            phrase = re.split(r"\b(?:to|towards|while|and)\b", phrase)[0]
            obstacle = choose(phrase, lambda obj: obj.kind != "zone")
            stage.append(f"avoid:{obstacle.name}")
        if "reach" in kinds:
            goal_text = re.split(r"\b(?:to|toward|towards|enter|reach)\b", text)[-1]
            goal = choose(goal_text, lambda obj: obj is not obstacle)
            stage.append(f"reach:{goal.name}")
        if not stage:
            if "backward" in text:
                stage = ["backward"]
            elif "spin" in text or "turn" in text:
                stage = ["spin"]
            elif "walk" in text or "forward" in text or "move" in text:
                stage = ["forward"]
            else:
                raise SpecError("This task stage has no measurable event")
        duration = re.search(r"(?:for|still)\s+(\d+(?:\.\d+)?)\s*(?:s\b|seconds?\b)", text)
        if duration:
            stage = [name + "@" + duration[1] if name.startswith(("stand_still", "stand_on:")) else name for name in stage]
        if ordered and len(stage) != 1:
            raise SpecError("Use one explicit measurable action per ordered stage in fake mode")
        events.extend(stage)
    metric = ("sequence:" + ">".join(events)) if ordered else ("all:" + "&".join(events) if len(events) > 1 else events[0])
    return validate_for_scene([{"metric": metric, "op": ">=", "value": 0.8},
                               {"metric": "scene_fell", "op": "<=", "value": 0.1}], request, world)


class FakeChecker:
    """Keyword rules, so offline runs get a plausible spec."""

    def write_spec(self, request: str, scene: scene.SceneSpec | None = None) -> list[dict[str, Any]]:
        planned = _scene_plan(request, scene)
        if planned is not None:
            return planned
        if requires_reach(request):
            return validate_for_scene([
                {"metric": "target_reached", "op": ">=", "value": 0.8},
                {"metric": "target_fell", "op": "<=", "value": 0.1},
            ], request, scene)
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
        return validate_for_scene(core + [NO_FALL], request, scene)


class FakeCritic:
    def review(self, request: str, result: RunResult, check: dict[str, Any] | None = None) -> Review:
        score = float(result.metrics.get("fake_quality", 5.0))
        return Review(score=score, critique=f"Fake critique for '{request}': try a stronger task term.")


class FakeRanker:
    def rank(self, request: str, snapshots: list[dict[str, Any]]) -> list[int]:
        """Highest episode reward first: fine offline, though rewards don't compare in real runs."""
        return sorted(range(len(snapshots)), key=lambda i: -(snapshots[i].get("episode_reward") or 0.0))
