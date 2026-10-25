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

Objects are spheres ("ball"), boxes, capsules (upright poles, or lying down with
`tilt`) and zones painted on the floor: the primitives MJX collides with each other
and with the floor. A fixed object can float (`elevation`, e.g. a hurdle bar); a
movable one can be launched (`speed`, aimed at the robot's start: a rolled ball).
`relevant` objects (the default) give the reward and the policy their features;
set it false for scenery.

`structures` are parametric terrain the robot interacts with as a whole: stairs,
ramp, wall, gap, field (of obstacles). Each compiles to a few static boxes or
capsules on one body, placed like an object, facing the robot (yaw 0 means the
robot walking straight at it goes up the stairs or ramp, or meets the wall
broadside). Its features describe one anchor point: the top landing of stairs
and ramps, the far platform of a gap, the centre of a wall, the far edge of a
field.

Claude only ever writes this JSON. scene_xml.py writes the MJCF from it, so no
model output reaches the XML: no includes, file paths, meshes or plugins.

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

KINDS = ("ball", "box", "zone", "capsule")
STRUCTURE_KINDS = ("stairs", "ramp", "wall", "gap", "field")
TERRAINS = ("flat", "rough")
MAX_OBJECTS = 6
MAX_STRUCTURES = 4
MAX_GEOMS = 24  # objects plus every structure's pieces; each one costs collision checks every step
MAX_RELEVANT = 6  # bodies with features in the observation
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
CAPSULE_RADIUS = (0.02, 0.4)
CAPSULE_LENGTH = (0.05, 2.0)  # the straight part between the two caps
MAX_ELEVATION = 1.0  # m; the lowest point of a fixed object above the floor
MAX_TILT = 90.0  # degrees of roll or pitch
MAX_SPEED = 5.0  # m/s; a launched movable object's starting speed
MAX_AIM = 90.0  # degrees off the line to the robot's start

NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,15}$")
# Object names become feature names (ball_dist, ...); these would clash with robot features.
RESERVED_NAMES = frozenset({"base", "target", "joint", "joints", "foot", "feet", "gravity", "up", "torques", "time", "bumper"})

OBJECT_KEYS = ("name", "kind", "color", "size", "distance", "bearing", "yaw", "movable", "mass",
               "relevant", "elevation", "tilt", "speed", "aim")
STRUCTURE_KEYS = ("name", "kind", "color", "distance", "bearing", "yaw", "relevant", "params")
SCENE_KEYS = ("terrain", "objects", "target", "goal", "structures")

# Structure parameters: name -> (default, low, high). Ints are counts. Defaults suit the Go1
# (legs about 0.2 m, so steps up to about 0.1 m and slopes up to about 15 degrees are climbable).
STRUCTURE_PARAMS: dict[str, dict[str, tuple[float, float, float]]] = {
    "stairs": {"steps": (4, 1, 8), "rise": (0.06, 0.02, 0.15), "run": (0.3, 0.2, 0.6), "width": (1.6, 0.6, 3.0),
               "landing": (1.0, 0.4, 2.0), "descend": (1, 0, 1)},
    "ramp": {"length": (2.0, 0.5, 4.0), "angle": (10.0, 2.0, 25.0), "width": (1.6, 0.6, 3.0),
             "landing": (1.0, 0.0, 2.0), "descend": (1, 0, 1)},
    "wall": {"length": (1.5, 0.2, 4.0), "height": (0.5, 0.03, 1.0), "thickness": (0.1, 0.05, 0.5)},
    "gap": {"width": (0.2, 0.05, 0.5), "height": (0.08, 0.03, 0.15), "length": (1.0, 0.5, 2.0),
            "span": (1.2, 0.6, 3.0)},
    "field": {"count": (6, 2, 12), "size": (0.1, 0.03, 0.4), "height": (0.5, 0.1, 1.0), "length": (3.0, 1.0, 6.0),
              "width": (2.0, 1.0, 4.0), "boxes": (0, 0, 1), "seed": (0, 0, 9999)},
}
INT_PARAMS = frozenset({"steps", "count", "boxes", "seed", "descend"})
FIELD_GAP = 0.5  # m of free floor between obstacles in a field, so the robot fits between them
STRUCTURE_HELP = {
    "stairs": "steps (count), rise and run (m per step), width (m), landing (m of flat top), "
              "descend (1: the same steps down the far side, so walking on doesn't drop off a ledge; 0: a ledge)",
    "ramp": "length (m, horizontal), angle (degrees), width (m), landing (m of flat top, 0 for none), "
            "descend (1: a matching slope down the far side; 0: a ledge)",
    "wall": "length (m, across the robot's path), height (m; low ones are hurdles), thickness (m)",
    "gap": "width (m of the gap), height (m of the two platforms), length (m of each platform), span (m across)",
    "field": "count (obstacles), size (radius or half side, m), height (m), length and width (m of the area), "
             "boxes (1 for boxes, 0 for poles), seed (layout)",
}
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
    relevant: bool = True  # gives the reward and the policy its features
    elevation: float = 0.0  # m of the lowest point above the floor; fixed objects only
    tilt: tuple[float, float] = (0.0, 0.0)  # degrees of (roll, pitch), fixed; a capsule at pitch 90 lies down
    speed: tuple[float, float] = (0.0, 0.0)  # m/s at reset, movable only: launched toward the robot's start
    aim: tuple[float, float] = (0.0, 0.0)  # degrees the launch is off the line to the robot's start, + = left

    @property
    def rgb(self) -> tuple[float, float, float]:
        return COLORS[self.color]

    @property
    def rotation(self) -> tuple[tuple[float, ...], ...]:
        """The tilt as a rotation matrix (roll about x, then pitch about y)."""
        return _tilt_matrix(*self.tilt)

    @property
    def half_height(self) -> float:
        """How far the object reaches below (and above) its centre, with its tilt."""
        r = self.rotation
        if self.kind == "ball":
            return self.size[0]
        if self.kind == "box":
            return sum(abs(r[2][i]) * 0.5 * self.size[i] for i in range(3))
        if self.kind == "capsule":
            return self.size[0] + 0.5 * self.size[1] * abs(r[2][2])
        return 0.0

    @property
    def footprint_radius(self) -> float:
        """Radius of the circle the object covers on the floor."""
        if self.kind == "box":
            if self.tilt == (0.0, 0.0):
                return 0.5 * math.hypot(self.size[0], self.size[1])
            return 0.5 * math.sqrt(sum(x * x for x in self.size))
        if self.kind == "capsule":
            r = self.rotation
            return self.size[0] + 0.5 * self.size[1] * math.hypot(r[0][2], r[1][2])
        return self.size[0]

    @property
    def rest_height(self) -> float:
        """z of the object's centre when it rests on flat ground (or floats at its elevation)."""
        if self.kind == "zone":
            return 0.0  # zones are a painted disc on the floor
        return self.elevation + self.half_height

    @property
    def top_height(self) -> float:
        """z of the object's highest point."""
        return 0.0 if self.kind == "zone" else self.elevation + 2 * self.half_height

    @property
    def reach_radius(self) -> float:
        """Robot base this close (horizontally, m) to the centre counts as having reached it.
        For a zone that means standing inside it; otherwise touching it with the front of the body."""
        if self.kind == "zone":
            return self.size[0]
        return self.footprint_radius + 0.35


@dataclass(frozen=True)
class Piece:
    """One geom of a structure, in the structure's frame: x away from the robot, y to its left."""

    kind: str  # box | capsule
    half: tuple[float, ...]  # box half sizes; capsule (radius, half length)
    pos: tuple[float, float, float]
    pitch: float = 0.0  # degrees about y; ramps
    ground: bool = True  # the robot can stand on its top (boxes)


@dataclass(frozen=True)
class Structure:
    name: str
    kind: str  # stairs | ramp | wall | gap | field
    color: str
    distance: tuple[float, float]  # m from the robot's start to the structure's near edge
    bearing: tuple[float, float]  # degrees from the robot's start heading, + = left
    yaw: tuple[float, float] = (0.0, 0.0)  # degrees turned from facing the robot squarely
    relevant: bool = True
    params: tuple[tuple[str, float], ...] = ()  # sorted (name, value) pairs; see STRUCTURE_PARAMS

    @property
    def rgb(self) -> tuple[float, float, float]:
        return COLORS[self.color]

    def p(self, key: str) -> float:
        return dict(self.params)[key]

    @property
    def pieces(self) -> tuple[Piece, ...]:
        return structure_pieces(self)

    @property
    def anchor(self) -> tuple[float, float, float]:
        """The point its features describe, in its frame (see the module docstring)."""
        if self.kind == "stairs":
            n, rise, run = self.p("steps"), self.p("rise"), self.p("run")
            return (n * run + 0.5 * self.p("landing"), 0.0, n * rise)
        if self.kind == "ramp":
            length, h = self.p("length"), self.p("length") * math.tan(math.radians(self.p("angle")))
            return (length + 0.5 * self.p("landing"), 0.0, h)
        if self.kind == "wall":
            return (0.5 * self.p("thickness"), 0.0, 0.5 * self.p("height"))
        if self.kind == "gap":
            length = self.p("length")
            return (1.5 * length + self.p("width"), 0.0, self.p("height"))
        return (self.p("length"), 0.0, 0.0)  # field: its far edge

    @property
    def top_height(self) -> float:
        return max(_piece_top(pc) for pc in self.pieces)

    @property
    def walkable(self) -> bool:
        """The robot is meant to walk on it (no trunk bumper: its body passes over the steps)."""
        return self.kind in ("stairs", "ramp", "gap")

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """(x0, x1, y0, y1) of its footprint in its own frame."""
        xs, ys = [], []
        for pc in self.pieces:
            hx, hy = (pc.half[0], pc.half[1]) if pc.kind == "box" else (pc.half[0], pc.half[0])
            if pc.pitch:
                hx = hx * math.cos(math.radians(pc.pitch)) + pc.half[2] * abs(math.sin(math.radians(pc.pitch)))
            xs += [pc.pos[0] - hx, pc.pos[0] + hx]
            ys += [pc.pos[1] - hy, pc.pos[1] + hy]
        return min(xs), max(xs), min(ys), max(ys)

    @property
    def reach_radius(self) -> float:
        """The base this close (horizontally, m) to the anchor counts as there."""
        if self.kind in ("stairs", "ramp"):
            return max(0.3, 0.5 * min(max(self.p("landing"), 0.4), self.p("width")))
        if self.kind == "gap":
            return 0.5 * min(self.p("length"), self.p("span"))
        return 0.5


@dataclass(frozen=True)
class SceneSpec:
    terrain: str = "flat"
    objects: tuple[SceneObject, ...] = field(default_factory=tuple)
    target: str | None = None  # the object (or structure) the target_* features refer to
    goal: str | None = None  # where the target must be moved to (needs a movable target)
    structures: tuple[Structure, ...] = field(default_factory=tuple)

    @property
    def is_empty(self) -> bool:
        """True for the plain flat floor, which is the stock Playground scene."""
        return self.terrain == "flat" and not self.objects and not self.structures

    @property
    def relevant_objects(self) -> tuple[SceneObject, ...]:
        return tuple(o for o in self.objects if o.relevant)

    @property
    def relevant_structures(self) -> tuple[Structure, ...]:
        return tuple(s for s in self.structures if s.relevant)

    @property
    def target_structure(self) -> Structure | None:
        return next((s for s in self.structures if s.name == self.target), None)

    @property
    def has_ground(self) -> bool:
        """Something to stand on above the floor: the terrain features (feet_ground_z, ...) exist."""
        return bool(self.structures) or any(o.kind == "box" and not o.movable for o in self.objects)

    @property
    def geom_count(self) -> int:
        return len(self.objects) + sum(len(s.pieces) for s in self.structures)

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
                {**asdict(o), "size": list(o.size), "distance": list(o.distance), "bearing": list(o.bearing),
                 "yaw": list(o.yaw), "tilt": list(o.tilt), "speed": list(o.speed), "aim": list(o.aim)}
                for o in self.objects
            ],
            "target": self.target,
            "goal": self.goal,
            "structures": [
                {"name": s.name, "kind": s.kind, "color": s.color, "distance": list(s.distance),
                 "bearing": list(s.bearing), "yaw": list(s.yaw), "relevant": s.relevant, "params": dict(s.params)}
                for s in self.structures
            ],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    def summary(self) -> str:
        """Plain-language layout, for the reward writer and the board."""
        ground = "rough, bumpy ground (bumps up to 5 cm)" if self.terrain == "rough" else "a flat floor"
        if not self.objects and not self.structures:
            return f"The robot is alone on {ground}."
        things = len(self.objects) + len(self.structures)
        lines = [f"The robot is on {ground} with {things} object(s), placed fresh each episode:"]
        for o in self.objects:
            lines.append(f"- {o.name}: {_object_phrase(o)}")
        for st in self.structures:
            lines.append(f"- {st.name}: {_structure_phrase(st)}")
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

    raw_structures = data.get("structures", [])
    structures: list[Structure] = []
    if not isinstance(raw_structures, list):
        errors.append("structures must be a list")
        raw_structures = []
    elif len(raw_structures) > MAX_STRUCTURES:
        errors.append(f"at most {MAX_STRUCTURES} structures, got {len(raw_structures)}")
        raw_structures = raw_structures[:MAX_STRUCTURES]
    for i, raw in enumerate(raw_structures):
        st, errs = _parse_structure(i, raw)
        errors += errs
        if st is not None:
            structures.append(st)

    names = [o.name for o in objects] + [st.name for st in structures]
    for name in sorted({n for n in names if names.count(n) > 1}):
        errors.append(f"object name {name!r} is used more than once; names must be unique")

    target = data.get("target")
    if target is None and names:
        target = names[0]
    if target is not None and target not in names:
        if not names:
            errors.append(f"target {target!r} names an object, but the scene has none")
        else:
            errors.append(f"target {target!r} is not an object name; use one of {', '.join(names)}")
    for o in objects + structures:
        if not o.relevant and o.name in (target, data.get("goal")):
            errors.append(f"{o.name!r} is the target or goal, so it must be relevant (its features are the task)")

    goal = _goal(data.get("goal"), target, objects, [o.name for o in objects], errors)

    relevant = sum(o.relevant for o in objects) + sum(st.relevant for st in structures)
    if relevant > MAX_RELEVANT:
        errors.append(f"at most {MAX_RELEVANT} relevant objects and structures, got {relevant}; "
                      "set relevant: false on scenery the robot doesn't need to sense")
    geoms = len(objects) + sum(len(st.pieces) for st in structures)
    if geoms > MAX_GEOMS:
        errors.append(f"the scene has {geoms} solid pieces (objects plus every step, platform and obstacle); "
                      f"at most {MAX_GEOMS}. Use fewer steps or field obstacles")

    if errors:
        raise SceneError("scene spec rejected:\n" + "\n".join(f"- {e}" for e in errors))
    return SceneSpec(terrain=terrain, objects=tuple(objects), target=target, goal=goal, structures=tuple(structures))


def _parse_structure(i: int, raw: Any) -> tuple[Structure | None, list[str]]:
    where = f"structures[{i}]"
    if not isinstance(raw, Mapping):
        return None, [f"{where} must be an object like {{\"name\": \"stairs\", \"kind\": \"stairs\", ...}}"]
    name = raw.get("name")
    if isinstance(name, str):
        where = f"structures[{i}] ({name})"
    errors: list[str] = []
    unknown = sorted(set(raw) - set(STRUCTURE_KEYS))
    if unknown:
        errors.append(f"{where}: unknown key(s) {', '.join(map(repr, unknown))}; allowed: {', '.join(STRUCTURE_KEYS)}")
    if not isinstance(name, str) or not NAME_RE.match(name):
        errors.append(f"{where}: name must be lowercase letters, digits or _, starting with a letter, "
                      f"at most 16 characters, got {name!r}")
    elif name in RESERVED_NAMES:
        errors.append(f"{where}: name {name!r} is reserved; pick another")
    kind = raw.get("kind")
    if kind not in STRUCTURE_KINDS:
        errors.append(f"{where}: kind must be one of {', '.join(STRUCTURE_KINDS)}, got {kind!r}")
        return None, errors
    color = raw.get("color", "gray")
    if not isinstance(color, str) or color not in COLORS:
        errors.append(f"{where}: color must be one of {', '.join(COLORS)}, got {color!r}")
    relevant = raw.get("relevant", True)
    if not isinstance(relevant, bool):
        errors.append(f"{where}: relevant must be true or false, got {relevant!r}")
        relevant = True

    table = STRUCTURE_PARAMS[kind]
    given = raw.get("params") or {}
    params: dict[str, float] = {}
    if not isinstance(given, Mapping):
        errors.append(f"{where}: params must be an object of {STRUCTURE_HELP[kind]}")
        given = {}
    extra = sorted(set(given) - set(table))
    if extra:
        errors.append(f"{where}: unknown {kind} param(s) {', '.join(map(repr, extra))}; "
                      f"a {kind} takes {STRUCTURE_HELP[kind]}")
    for key, (default, lo, hi) in table.items():
        value = given.get(key, default)
        if not _is_number(value) or (key in INT_PARAMS and float(value) != int(value)):
            errors.append(f"{where}: {key} must be {'a whole number' if key in INT_PARAMS else 'a number'}, got {value!r}")
            continue
        if not lo <= value <= hi:
            errors.append(f"{where}: {key} must be between {lo:g} and {hi:g}, got {value:g}")
            continue
        params[key] = float(int(value)) if key in INT_PARAMS else float(value)

    distance = _range(where, "distance", raw.get("distance"), (0.0, MAX_DISTANCE), "m", errors)
    bearing = _range(where, "bearing", raw.get("bearing", 0.0), (-180.0, 180.0), "degrees", errors)
    yaw = _range(where, "yaw", raw.get("yaw", 0.0), (-180.0, 180.0), "degrees", errors)
    if errors:
        return None, errors
    st = Structure(name=name, kind=kind, color=color, distance=distance, bearing=bearing, yaw=yaw,
                   relevant=relevant, params=tuple(sorted(params.items())))
    if kind == "field" and len(st.pieces) < st.p("count"):
        return None, [f"{where}: only {len(st.pieces)} of {st.p('count'):g} obstacles fit in "
                      f"{st.p('length'):g} x {st.p('width'):g} m with {FIELD_GAP} m between them; "
                      "make the field bigger, or the obstacles fewer or smaller"]
    near = _structure_clearance(st)
    if near < ROBOT_CLEARANCE:
        return None, [f"{where}: the {kind} comes within {near:.2f} m of the robot's start (it needs "
                      f"{ROBOT_CLEARANCE} m so nothing spawns on the robot); increase distance"]
    return st, []


def _structure_clearance(st: Structure) -> float:
    """Closest the structure's footprint gets to the robot's start, over its placement ranges."""
    x0, x1, y0, y1 = st.bounds
    best = math.inf
    yaws = {st.yaw[0], st.yaw[1]} | ({0.0} if st.yaw[0] <= 0 <= st.yaw[1] else set())
    for yaw in yaws:
        a = math.radians(yaw)
        c, s_ = math.cos(a), math.sin(a)
        # The robot's start in the structure's frame: d metres in front of its origin, turned by -yaw.
        d = st.distance[0]
        px, py = -d * c, d * s_
        dx = max(x0 - px, 0.0, px - x1)
        dy = max(y0 - py, 0.0, py - y1)
        best = min(best, math.hypot(dx, dy))
    return best


def _tilt_matrix(roll: float, pitch: float) -> tuple[tuple[float, ...], ...]:
    """R = Ry(pitch) @ Rx(roll), degrees."""
    a, b = math.radians(roll), math.radians(pitch)
    ca, sa, cb, sb = math.cos(a), math.sin(a), math.cos(b), math.sin(b)
    return ((cb, sb * sa, sb * ca), (0.0, ca, -sa), (-sb, cb * sa, cb * ca))


def _piece_top(pc: Piece) -> float:
    if pc.kind == "capsule":
        return pc.pos[2] + pc.half[0] + pc.half[1]
    b = math.radians(pc.pitch)
    return pc.pos[2] + pc.half[2] * math.cos(b) + pc.half[0] * abs(math.sin(b))


def structure_pieces(st: Structure) -> tuple[Piece, ...]:
    """The geoms a structure compiles to, in its frame (x away from the robot, z up from the floor)."""
    p = st.p
    if st.kind == "stairs":
        n, rise, run, half_w = int(p("steps")), p("rise"), p("run"), 0.5 * p("width")
        end = n * run + p("landing") + (n * run if p("descend") else 0.0)
        pieces = []
        for i in range(n):  # step i is a solid block from its riser to the far end (or its mirror), so nothing floats
            x0, x1 = i * run, end - (i * run if p("descend") else 0.0)
            h = (i + 1) * rise
            pieces.append(Piece("box", (0.5 * (x1 - x0), half_w, 0.5 * h), (0.5 * (x0 + x1), 0.0, 0.5 * h)))
        return tuple(pieces)
    if st.kind == "ramp":
        length, a = p("length"), math.radians(p("angle"))
        h, half_w, t = length * math.tan(a), 0.5 * p("width"), 0.05
        slope = length / math.cos(a)
        # A thin slab whose top face runs from the floor at x=0 to height h at x=length.
        cx, cz = 0.5 * length + t * math.sin(a), 0.5 * h - t * math.cos(a)
        pieces = [Piece("box", (0.5 * slope, half_w, t), (cx, 0.0, cz), pitch=-p("angle"))]
        if p("landing") > 0:
            pieces.append(Piece("box", (0.5 * p("landing"), half_w, 0.5 * h), (length + 0.5 * p("landing"), 0.0, 0.5 * h)))
        if p("descend"):  # the mirror image, from the landing's far edge back to the floor
            far = 2 * length + p("landing")
            pieces.append(Piece("box", (0.5 * slope, half_w, t), (far - cx, 0.0, cz), pitch=p("angle")))
        return tuple(pieces)
    if st.kind == "wall":
        t, half_l, h = p("thickness"), 0.5 * p("length"), p("height")
        return (Piece("box", (0.5 * t, half_l, 0.5 * h), (0.5 * t, 0.0, 0.5 * h)),)
    if st.kind == "gap":
        length, gap, h, half_s = p("length"), p("width"), p("height"), 0.5 * p("span")
        return (Piece("box", (0.5 * length, half_s, 0.5 * h), (0.5 * length, 0.0, 0.5 * h)),
                Piece("box", (0.5 * length, half_s, 0.5 * h), (1.5 * length + gap, 0.0, 0.5 * h)))
    # field: obstacles scattered over [0, length] x [-width/2, width/2], the same layout every episode
    import random

    rng = random.Random(int(p("seed")))
    size, h, boxes = p("size"), p("height"), bool(p("boxes"))
    spacing = 2 * size * (math.sqrt(2) if boxes else 1.0) + FIELD_GAP
    spots: list[tuple[float, float]] = []
    for _ in range(400):
        if len(spots) == int(p("count")):
            break
        x = rng.uniform(size, p("length") - size)
        y = rng.uniform(-0.5 * p("width") + size, 0.5 * p("width") - size)
        if all(math.hypot(x - a, y - b) >= spacing for a, b in spots):
            spots.append((x, y))
    if boxes:
        return tuple(Piece("box", (size, size, 0.5 * h), (x, y, 0.5 * h), ground=False) for x, y in spots)
    return tuple(Piece("capsule", (size, max(0.5 * h - size, 0.01)), (x, y, 0.5 * h), ground=False) for x, y in spots)


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
    elif kind == "capsule":
        errors += _check_size(where, size, [("radius", CAPSULE_RADIUS), ("length", CAPSULE_LENGTH)], "[radius, length]")

    relevant = raw.get("relevant", True)
    if not isinstance(relevant, bool):
        errors.append(f"{where}: relevant must be true or false, got {relevant!r}")
        relevant = True
    elevation = raw.get("elevation", 0.0)
    if not _is_number(elevation) or not 0.0 <= elevation <= MAX_ELEVATION:
        errors.append(f"{where}: elevation must be between 0 and {MAX_ELEVATION} m, got {elevation!r}")
        elevation = 0.0
    elif elevation and (movable or kind == "zone"):
        errors.append(f"{where}: only fixed solid objects can float (elevation > 0); a movable one would fall")
    tilt = _numbers(raw.get("tilt", [0.0, 0.0]))
    if tilt is None or len(tilt) != 2 or any(abs(t) > MAX_TILT for t in tilt):
        errors.append(f"{where}: tilt must be [roll, pitch] in degrees, each within +-{MAX_TILT:g}, got {raw.get('tilt')!r}")
        tilt = [0.0, 0.0]
    elif any(tilt) and kind in ("ball", "zone"):
        errors.append(f"{where}: a {kind} can't be tilted")
    speed = _range(where, "speed", raw.get("speed", 0.0), (0.0, MAX_SPEED), "m/s", errors)
    aim = _range(where, "aim", raw.get("aim", 0.0), (-MAX_AIM, MAX_AIM), "degrees", errors)
    if speed[1] > 0 and not movable:
        errors.append(f"{where}: only movable objects can be launched (speed > 0); set movable: true")

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
        bearing=bearing, yaw=yaw, movable=movable, mass=float(mass), relevant=relevant,
        elevation=float(elevation), tilt=(float(tilt[0]), float(tilt[1])), speed=speed, aim=aim,
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
    elif o.kind == "capsule":
        lying = abs(o.tilt[0]) >= 45 or abs(o.tilt[1]) >= 45
        shape = (f"{o.color} {'bar lying down' if lying else 'pole'}, radius {o.size[0]:g} m, "
                 f"{o.size[1] + 2 * o.size[0]:g} m long")
    else:
        shape = f"{o.color} zone painted on the floor, radius {o.size[0]:g} m"
    if o.elevation:
        shape += f", floating {o.elevation:g} m above the floor (its top at {o.top_height:.2f} m)"
    how = (f"movable ({o.mass:g} kg, the robot's feet can push it)" if o.movable
           else ("fixed in place" if o.kind != "zone" else "flat, the robot can walk onto it"))
    if o.speed[1] > 0:
        how += f", launched at {_span(o.speed, ' m/s')} toward the robot's start ({_span(o.aim, ' deg')} off)"
    if not o.relevant:
        how += "; scenery (the robot doesn't sense it)"
    return (f"{shape}; {how}; starts {_span(o.distance, ' m')} from the robot, "
            f"{_span(o.bearing, ' deg')} from its starting heading (+ = left)")


def _structure_phrase(st: Structure) -> str:
    p = st.p
    if st.kind == "stairs":
        shape = (f"{st.color} stairs: {p('steps'):g} steps of {p('rise'):g} m rise and {p('run'):g} m run, "
                 f"{p('width'):g} m wide, then a {p('landing'):g} m landing {st.top_height:.2f} m up"
                 + (", and the same steps down the far side" if p("descend") else ", ending in a ledge"))
    elif st.kind == "ramp":
        shape = (f"{st.color} ramp: {p('length'):g} m long at {p('angle'):g} deg, {p('width'):g} m wide, "
                 f"rising to {st.top_height:.2f} m" + (f", then a {p('landing'):g} m landing" if p("landing") else "")
                 + (", then a slope back down" if p("descend") else ", ending in a ledge"))
    elif st.kind == "wall":
        shape = f"{st.color} wall {p('length'):g} m long, {p('height'):g} m tall, {p('thickness'):g} m thick, across the path"
    elif st.kind == "gap":
        shape = (f"{st.color} gap: two {p('height'):g} m high platforms, {p('length'):g} m long each, "
                 f"with a {p('width'):g} m gap between them")
    else:
        shape = (f"{st.color} field of {len(st.pieces)} {'boxes' if p('boxes') else 'poles'} "
                 f"({p('height'):g} m tall) over {p('length'):g} x {p('width'):g} m")
    anchor = {"stairs": "the top landing", "ramp": "the top", "wall": "the wall's centre",
              "gap": "the far platform", "field": "the far edge of the field"}[st.kind]
    sensed = f"its features are at {anchor}" if st.relevant else "scenery (the robot doesn't sense it)"
    return (f"{shape}; fixed; its near edge starts {_span(st.distance, ' m')} from the robot, "
            f"{_span(st.bearing, ' deg')} from its starting heading, facing the robot"
            f"{'' if st.yaw == (0.0, 0.0) else ' turned ' + _span(st.yaw, ' deg')}; {sensed}")
