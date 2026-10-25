"""Scene spec: the objects and terrain around the robot, as small validated JSON.

The scene writer (Claude) outputs a spec like this; scene_xml.py compiles it
into the Go1 MuJoCo scene. Nothing here imports MuJoCo or JAX, so the server,
fakes and prompts can use it.

    {
      "terrain": "flat",
      "objects": [
        {"name": "ball", "kind": "ball", "color": "red", "size": [0.15],
         "distance": [1.5, 3.0], "bearing": [-60, 60], "yaw": [0, 0],
         "movable": false, "mass": 0.5}
      ],
      "target": "ball"
    }

For a "move the object somewhere" task, `goal` names where the target must end up
(usually a zone): {"target": "ball", "goal": "goal_zone"}. When a movable target
and exactly one zone are given without a goal, the zone is taken as the goal.

Objects are placed relative to where the robot starts each episode: `distance`
metres away, at `bearing` degrees from its starting heading (positive = to its
left). Each range is drawn fresh at every reset, so a policy learns the skill
rather than one spot. A number instead of a [low, high] pair fixes the value.

parse() raises SceneError with every problem listed, written for the scene
writer to fix, the same way the reward gate reports reward bugs.
"""

import json
import math
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any

KINDS = ("ball", "box", "zone")
TERRAINS = ("flat", "rough")
MAX_OBJECTS = 6
MAX_DISTANCE = 8.0  # m from the robot's start; keeps objects on the floor and near the camera
ROBOT_CLEARANCE = 0.5  # m between the robot's start and an object's edge, so nothing spawns on it

COLORS: dict[str, tuple[float, float, float]] = {
    "red": (0.86, 0.16, 0.16),
    "green": (0.18, 0.70, 0.25),
    "blue": (0.16, 0.36, 0.86),
    "yellow": (0.95, 0.80, 0.10),
    "orange": (0.96, 0.50, 0.10),
    "purple": (0.55, 0.25, 0.75),
    "pink": (0.95, 0.45, 0.70),
    "cyan": (0.10, 0.75, 0.85),
    "brown": (0.50, 0.32, 0.18),
    "white": (0.92, 0.92, 0.92),
    "gray": (0.50, 0.50, 0.50),
    "black": (0.08, 0.08, 0.08),
}

# Limits per kind: (min, max) metres. ball and zone sizes are [radius]; box is [length, width, height].
BALL_RADIUS = (0.05, 0.5)
ZONE_RADIUS = (0.2, 2.0)
BOX_SIDE = (0.05, 2.0)
BOX_HEIGHT = (0.03, 1.0)
MOVABLE_BOX_SIDE = 1.0  # a pushable box bigger than this is a wall the robot can't move
MASS = (0.05, 20.0)  # kg, movable objects only

NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,15}$")
# Object names become feature names (ball_dist, ...); these would clash with robot features.
RESERVED_NAMES = frozenset({"base", "target", "joint", "joints", "foot", "feet", "gravity", "up", "torques", "time", "bumper"})

OBJECT_KEYS = ("name", "kind", "color", "size", "distance", "bearing", "yaw", "movable", "mass")
SCENE_KEYS = ("terrain", "objects", "target", "goal")
GOAL_MARGIN = 0.05  # m; a solid goal counts as reached when the target touches it, plus this


class SceneError(ValueError):
    """The scene spec is unusable. The message is meant for the scene writer."""


@dataclass(frozen=True)
class SceneObject:
    name: str
    kind: str  # ball | box | zone
    color: str
    size: tuple[float, ...]  # ball, zone: (radius,); box: (length, width, height)
    distance: tuple[float, float]  # m from the robot's start position
    bearing: tuple[float, float]  # degrees from the robot's start heading, + = left
    yaw: tuple[float, float] = (0.0, 0.0)  # degrees, the object's turn relative to the robot's start heading
    movable: bool = False  # True: a free body the robot can push; False: fixed in place
    mass: float = 1.0  # kg, used when movable

    @property
    def rgb(self) -> tuple[float, float, float]:
        return COLORS[self.color]

    @property
    def footprint_radius(self) -> float:
        """Radius of the circle the object covers on the floor."""
        if self.kind == "box":
            return 0.5 * math.hypot(self.size[0], self.size[1])
        return self.size[0]

    @property
    def rest_height(self) -> float:
        """z of the object's centre when it rests on flat ground."""
        if self.kind == "ball":
            return self.size[0]
        if self.kind == "box":
            return 0.5 * self.size[2]
        return 0.0  # zones are a painted disc on the floor

    @property
    def reach_radius(self) -> float:
        """Robot base this close (horizontally, m) to the centre counts as having reached it.
        For a zone that means standing inside it; otherwise touching it with the front of the body."""
        if self.kind == "zone":
            return self.size[0]
        return self.footprint_radius + 0.35


@dataclass(frozen=True)
class SceneSpec:
    terrain: str = "flat"
    objects: tuple[SceneObject, ...] = field(default_factory=tuple)
    target: str | None = None  # the object the target_* features refer to
    goal: str | None = None  # where the target must be moved to (needs a movable target)

    @property
    def is_empty(self) -> bool:
        """True for the plain flat floor, which is the stock Playground scene."""
        return self.terrain == "flat" and not self.objects

    @property
    def target_object(self) -> SceneObject | None:
        return next((o for o in self.objects if o.name == self.target), None)

    @property
    def goal_object(self) -> SceneObject | None:
        return next((o for o in self.objects if o.name == self.goal), None)

    @property
    def goal_radius(self) -> float | None:
        """The target is in the goal when its centre is this close (horizontally, m) to the goal's:
        inside a zone, or touching a solid goal."""
        target, goal = self.target_object, self.goal_object
        if target is None or goal is None:
            return None
        if goal.kind == "zone":
            return goal.size[0]
        return goal.footprint_radius + target.footprint_radius + GOAL_MARGIN

    def to_dict(self) -> dict[str, Any]:
        return {
            "terrain": self.terrain,
            "objects": [
                {**asdict(o), "size": list(o.size), "distance": list(o.distance),
                 "bearing": list(o.bearing), "yaw": list(o.yaw)}
                for o in self.objects
            ],
            "target": self.target,
            "goal": self.goal,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    def summary(self) -> str:
        """Plain-language layout, for the reward writer and the board."""
        ground = "rough, bumpy ground (bumps up to 5 cm)" if self.terrain == "rough" else "a flat floor"
        if not self.objects:
            return f"The robot is alone on {ground}."
        lines = [f"The robot is on {ground} with {len(self.objects)} object(s), placed fresh each episode:"]
        for o in self.objects:
            lines.append(f"- {o.name}: {_object_phrase(o)}")
        if self.target:
            lines.append(f"The target object is {self.target}.")
        if self.goal:
            lines.append(f"The task is to move {self.target} to {self.goal}: it counts as there when its centre "
                         f"is within {self.goal_radius:.2f} m of {self.goal}'s centre.")
        return "\n".join(lines)


EMPTY = SceneSpec()


def parse(data: Mapping[str, Any] | str | None) -> SceneSpec:
    """Validate a scene spec (a dict or JSON text). None or {} means the empty flat scene."""
    if data is None:
        return EMPTY
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except json.JSONDecodeError as exc:
            raise SceneError(f"scene spec is not valid JSON: {exc}") from exc
    if not isinstance(data, Mapping):
        raise SceneError(f"scene spec must be a JSON object, got {type(data).__name__}")

    errors: list[str] = []
    unknown = sorted(set(data) - set(SCENE_KEYS))
    if unknown:
        errors.append(f"unknown key(s) {', '.join(map(repr, unknown))}; allowed: {', '.join(SCENE_KEYS)}")

    terrain = data.get("terrain", "flat")
    if terrain not in TERRAINS:
        errors.append(f"terrain must be one of {', '.join(TERRAINS)}, got {terrain!r}")

    raw_objects = data.get("objects", [])
    objects: list[SceneObject] = []
    if not isinstance(raw_objects, list):
        errors.append("objects must be a list")
        raw_objects = []
    elif len(raw_objects) > MAX_OBJECTS:
        errors.append(f"at most {MAX_OBJECTS} objects, got {len(raw_objects)}; keep only what the behavior needs")
        raw_objects = raw_objects[:MAX_OBJECTS]
    for i, raw in enumerate(raw_objects):
        obj, errs = _parse_object(i, raw)
        errors += errs
        if obj is not None:
            objects.append(obj)

    names = [o.name for o in objects]
    for name in sorted({n for n in names if names.count(n) > 1}):
        errors.append(f"object name {name!r} is used more than once; names must be unique")

    target = data.get("target")
    if target is None and objects:
        target = objects[0].name
    if target is not None and target not in names:
        if not objects:
            errors.append(f"target {target!r} names an object, but the scene has none")
        else:
            errors.append(f"target {target!r} is not an object name; use one of {', '.join(names)}")

    goal = _goal(data.get("goal"), target, objects, names, errors)

    if errors:
        raise SceneError("scene spec rejected:\n" + "\n".join(f"- {e}" for e in errors))
    return SceneSpec(terrain=terrain, objects=tuple(objects), target=target, goal=goal)


def _goal(goal: Any, target: str | None, objects: list[SceneObject], names: list[str], errors: list[str]) -> str | None:
    """The goal object's name, or None. Inferred for a movable target and exactly one zone."""
    by_name = {o.name: o for o in objects}
    moved = by_name.get(target) if target else None
    if goal in (None, ""):
        zones = [o.name for o in objects if o.kind == "zone"]
        return zones[0] if moved is not None and moved.movable and len(zones) == 1 else None
    if goal not in names:
        errors.append(f"goal {goal!r} is not an object name; use one of {', '.join(names) or '(no objects)'}")
    elif goal == target:
        errors.append(f"goal {goal!r} is the target itself; the goal is where the target must be moved to")
    elif moved is not None and not moved.movable:
        errors.append(f"goal {goal!r} needs a movable target, but {target!r} is fixed; set movable: true on {target!r}")
    elif by_name[goal].movable:
        errors.append(f"goal {goal!r} is movable; a goal must stay put (use a zone or a fixed object)")
    return goal


def _parse_object(i: int, raw: Any) -> tuple[SceneObject | None, list[str]]:
    where = f"objects[{i}]"
    if not isinstance(raw, Mapping):
        return None, [f"{where} must be an object like {{\"name\": \"ball\", \"kind\": \"ball\", ...}}"]
    name = raw.get("name")
    if isinstance(name, str):
        where = f"objects[{i}] ({name})"
    errors: list[str] = []
    unknown = sorted(set(raw) - set(OBJECT_KEYS))
    if unknown:
        errors.append(f"{where}: unknown key(s) {', '.join(map(repr, unknown))}; allowed: {', '.join(OBJECT_KEYS)}")

    if not isinstance(name, str) or not NAME_RE.match(name):
        errors.append(f"{where}: name must be lowercase letters, digits or _, starting with a letter, "
                      f"at most 16 characters (e.g. \"red_ball\"), got {name!r}")
    elif name in RESERVED_NAMES:
        errors.append(f"{where}: name {name!r} is reserved for robot features or internal geometry; pick another")

    kind = raw.get("kind")
    if kind not in KINDS:
        errors.append(f"{where}: kind must be one of {', '.join(KINDS)}, got {kind!r}")

    color = raw.get("color", "gray")
    if not isinstance(color, str) or color not in COLORS:
        errors.append(f"{where}: color must be one of {', '.join(COLORS)}, got {color!r}")

    movable = raw.get("movable", False)
    if not isinstance(movable, bool):
        errors.append(f"{where}: movable must be true or false, got {movable!r}")
        movable = False
    if movable and kind == "zone":
        errors.append(f"{where}: a zone is a patch of floor and cannot be movable")

    size = _numbers(raw.get("size"))
    if kind == "ball":
        errors += _check_size(where, size, [("radius", BALL_RADIUS)], "[radius]")
    elif kind == "zone":
        errors += _check_size(where, size, [("radius", ZONE_RADIUS)], "[radius]")
    elif kind == "box":
        side_max = MOVABLE_BOX_SIDE if movable else BOX_SIDE[1]
        dims = [("length", (BOX_SIDE[0], side_max)), ("width", (BOX_SIDE[0], side_max)), ("height", BOX_HEIGHT)]
        errors += _check_size(where, size, dims, "[length, width, height]")

    mass = raw.get("mass", 1.0)
    if not _is_number(mass):
        errors.append(f"{where}: mass must be a number of kg, got {mass!r}")
        mass = 1.0
    elif movable and not MASS[0] <= mass <= MASS[1]:
        errors.append(f"{where}: mass must be between {MASS[0]} and {MASS[1]} kg, got {mass}")

    distance = _range(where, "distance", raw.get("distance"), (0.0, MAX_DISTANCE), "m", errors)
    bearing = _range(where, "bearing", raw.get("bearing", 0.0), (-180.0, 180.0), "degrees", errors)
    yaw = _range(where, "yaw", raw.get("yaw", 0.0), (-180.0, 180.0), "degrees", errors)

    if errors:
        return None, errors
    obj = SceneObject(
        name=name, kind=kind, color=color, size=tuple(float(s) for s in size), distance=distance,
        bearing=bearing, yaw=yaw, movable=movable, mass=float(mass),
    )
    # A zone may sit under the robot (e.g. "stay inside the circle"); solid objects may not.
    if obj.kind != "zone":
        nearest = ROBOT_CLEARANCE + obj.footprint_radius
        if obj.distance[0] < nearest:
            return None, [f"{where}: distance must start at {nearest:.2f} m or more so the {kind} "
                          f"does not spawn on the robot, got {obj.distance[0]}"]
    return obj, []


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _numbers(x: Any) -> list[float] | None:
    if _is_number(x):
        return [float(x)]
    if isinstance(x, list) and x and all(_is_number(v) for v in x):
        return [float(v) for v in x]
    return None


def _check_size(where: str, size: list[float] | None, dims: list[tuple[str, tuple[float, float]]], form: str) -> list[str]:
    if size is None or len(size) != len(dims):
        return [f"{where}: size must be {form} in metres, got {size!r}"]
    errs = []
    for value, (label, (lo, hi)) in zip(size, dims):
        if not lo <= value <= hi:
            errs.append(f"{where}: {label} must be between {lo} and {hi} m, got {value}")
    return errs


def _range(
    where: str, key: str, x: Any, bounds: tuple[float, float], unit: str, errors: list[str]
) -> tuple[float, float]:
    """A number or a [low, high] pair, inside bounds. Appends to errors and returns (0, 0) if bad."""
    values = _numbers(x)
    if values is None or len(values) not in (1, 2):
        errors.append(f"{where}: {key} must be a number or a [low, high] pair in {unit}, got {x!r}")
        return (0.0, 0.0)
    lo, hi = values[0], values[-1]
    if lo > hi:
        errors.append(f"{where}: {key} range [{lo}, {hi}] has low above high")
    elif lo < bounds[0] or hi > bounds[1]:
        errors.append(f"{where}: {key} must stay within [{bounds[0]:g}, {bounds[1]:g}] {unit}, got [{lo:g}, {hi:g}]")
    return (lo, hi)


def _span(r: tuple[float, float], unit: str) -> str:
    return f"{r[0]:g}{unit}" if r[0] == r[1] else f"{r[0]:g} to {r[1]:g}{unit}"


def _object_phrase(o: SceneObject) -> str:
    if o.kind == "box":
        shape = f"{o.color} box {o.size[0]:g} x {o.size[1]:g} m, {o.size[2]:g} m tall"
    elif o.kind == "ball":
        shape = f"{o.color} ball, radius {o.size[0]:g} m"
    else:
        shape = f"{o.color} zone painted on the floor, radius {o.size[0]:g} m"
    how = (f"movable ({o.mass:g} kg, the robot's feet can push it)" if o.movable
           else ("fixed in place" if o.kind != "zone" else "flat, the robot can walk onto it"))
    return (f"{shape}; {how}; starts {_span(o.distance, ' m')} from the robot, "
            f"{_span(o.bearing, ' deg')} from its starting heading (+ = left)")
