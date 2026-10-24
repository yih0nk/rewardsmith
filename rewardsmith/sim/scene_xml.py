"""Compile a SceneSpec into the MJCF scene the Go1 env loads.

Starts from Playground's feet-only Go1 scene (flat or rough) and adds one body
per object. Fixed objects are mocap bodies and movable ones are free bodies,
so reset can move both every episode without recompiling. Free bodies come
after the robot in the body tree, so the robot keeps qpos[:19] and qvel[:18].

Collision groups follow the Go1 scene: the floor has contype=1/conaffinity=0
and the feet contype=0/conaffinity=1. Solid objects get contype=3 (bits 1 and 2):
fixed ones conaffinity=0 (they touch feet, the bumper and movable objects, not
the floor), movable ones conaffinity=1 (they also rest on the floor). Zones are
painted on the floor and touch nothing. With tall objects the robot also gets a hidden
"bumper" box on its trunk with conaffinity=2, so its body is blocked by them
(instead of walking through and climbing over them) but never touches the
floor, which would change the stock locomotion. The bumper hangs from the
trunk to 10 cm above the floor, so a ball meets a flat face rather than
ending up under the belly as a fulcrum. Objects lower than BUMPER_MIN_TOP
skip bit 2: they are steps, which the feet climb and the body passes over.

The rendered video uses this model, so objects are visible geoms (group 0).
The scene also adds an "overview" camera that looks at the robot from high up,
wide enough to show objects a few metres away.
"""

import hashlib
import math
import os
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

from .scene import SceneObject, SceneSpec

PREFIX = "obj_"  # body, geom and material names: obj_<name>, so they can't clash with the robot's
ROUGH_BUMP = 0.05  # max height of Playground's rough heightfield, m
OVERVIEW_CAMERA = "overview"
ROBOT_FILE = "go1_mjx_feetonly.xml"
BUMPER = "obj_bumper"
BUMPER_HALF_SIZE = (0.27, 0.10, 0.12)  # m; trunk plus hips and head, standing: 0.10 to 0.34 m above the floor
BUMPER_POS = (0.03, 0.0, -0.06)  # from the trunk centre
BUMPER_MIN_TOP = 0.15  # m; objects whose top is lower don't touch the bumper

# Contact budget for the warp backend, which preallocates contacts and constraint rows.
# Playground's flat Go1 uses 4 contacts per env (the feet) and 40 constraint rows.
BASE_CONTACTS, BASE_ROWS = 4, 40
ROUGH_CONTACTS, ROUGH_ROWS = 8, 60
ROWS_PER_CONTACT = 4  # condim 3, pyramidal friction cone
ENVS_PER_BUDGET = 8192  # Playground sizes naconmax for this many envs


def body_name(obj: SceneObject) -> str:
    return PREFIX + obj.name


def contact_budget(spec: SceneSpec) -> tuple[int, int]:
    """(naconmax, njmax) for the env config: room for each object's contacts."""
    contacts, rows = (ROUGH_CONTACTS, ROUGH_ROWS) if spec.terrain == "rough" else (BASE_CONTACTS, BASE_ROWS)
    extra = 0
    for o in spec.objects:
        if o.kind == "zone":
            continue
        extra += 4 + (2 if _hits_bumper(o) else 0)  # up to four feet, and the bumper
        if o.movable:
            extra += 4 if o.kind == "box" else 1  # resting on the floor
    return (contacts + extra) * ENVS_PER_BUDGET, rows + ROWS_PER_CONTACT * extra


def compile_xml(spec: SceneSpec, base_xml: str, sensor_xml: str, robot_xml: str | None = None) -> str:
    """Return the base Go1 scene with spec's objects added.

    base_xml is Playground's scene file (flat or rough) and sensor_xml its
    sensor_feet.xml, which is inlined so foot contact can count objects too.
    robot_xml is the Go1 file the scene includes; with solid objects it is
    inlined to give the trunk its bumper.
    """
    root = ET.fromstring(base_xml)
    if robot_xml is not None and any(_hits_bumper(o) for o in spec.objects):
        _inline_robot_with_bumper(root, robot_xml)
    asset = _child(root, "asset")
    world = _child(root, "worldbody")
    for o in spec.objects:
        r, g, b = o.rgb
        alpha = 0.55 if o.kind == "zone" else 1.0
        ET.SubElement(asset, "material", name=body_name(o), rgba=f"{r:.3f} {g:.3f} {b:.3f} {alpha}",
                      specular="0.3", shininess="0.3")
        world.append(_body(o, _nominal_pos(o, spec.terrain)))

    ET.SubElement(world, "camera", name=OVERVIEW_CAMERA, mode="targetbodycom", target="trunk",
                  pos="0 -4.5 4.5", fovy="55")

    _inline_sensors(root, sensor_xml, any_contact=any(o.kind != "zone" for o in spec.objects))
    _extend_keyframes(root, [o for o in spec.objects if o.movable], spec.terrain)
    return ET.tostring(root, encoding="unicode")


def write_scene_xml(spec: SceneSpec) -> Path:
    """Compile spec against Playground's Go1 files and write it to a cache file; return its path.

    Go1's base env reads its scene from a path, and includes resolve through
    its asset dict, so the file can live anywhere. Named by content hash, so
    parallel training processes share one file.
    """
    from mujoco_playground._src.locomotion.go1 import go1_constants as consts

    task = "rough_terrain" if spec.terrain == "rough" else "flat_terrain"
    base = consts.task_to_xml(task)
    xml = compile_xml(
        spec, base.read_text(), (base.parent / "sensor_feet.xml").read_text(), (base.parent / ROBOT_FILE).read_text()
    )
    out_dir = Path(os.environ.get("REWARDSMITH_SCENE_CACHE", Path(tempfile.gettempdir()) / "rewardsmith_scenes"))
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"scene_{hashlib.sha256(xml.encode()).hexdigest()[:16]}.xml"
    if not path.exists():
        tmp = path.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(xml)
        tmp.replace(path)  # atomic, so a concurrent reader never sees half a file
    return path


def _child(root: ET.Element, tag: str) -> ET.Element:
    el = root.find(tag)
    if el is None:
        el = ET.SubElement(root, tag)
    return el


def spawn_z(o: SceneObject, terrain: str) -> float:
    """Height of the object's centre at reset. Movable objects on rough ground start above
    the bumps and drop; fixed ones sit at floor level, partly sunk into any bump."""
    lift = ROUGH_BUMP if terrain == "rough" else 0.0
    if o.kind == "zone":
        return lift + 0.003
    return o.rest_height + (lift if o.movable else 0.0)


def _nominal_pos(o: SceneObject, terrain: str) -> tuple[float, float, float]:
    """Mid-range placement for a robot at the origin facing +x. Reset replaces it every episode."""
    d = 0.5 * (o.distance[0] + o.distance[1])
    a = math.radians(0.5 * (o.bearing[0] + o.bearing[1]))
    return d * math.cos(a), d * math.sin(a), spawn_z(o, terrain)


def _fmt(*xs: float) -> str:
    return " ".join(f"{x:.4g}" for x in xs)


def _hits_bumper(o: SceneObject) -> bool:
    top = 2 * o.size[0] if o.kind == "ball" else (o.size[2] if o.kind == "box" else 0.0)
    return top >= BUMPER_MIN_TOP


def _body(o: SceneObject, pos: tuple[float, float, float]) -> ET.Element:
    name = body_name(o)
    body = ET.Element("body", name=name, pos=_fmt(*pos))
    geom = {"name": name, "material": name, "group": "0"}
    if o.kind == "ball":
        geom.update(type="sphere", size=_fmt(o.size[0]))
    elif o.kind == "box":
        geom.update(type="box", size=_fmt(*(0.5 * s for s in o.size)))
    else:
        geom.update(type="cylinder", size=_fmt(o.size[0], 0.003))

    contype = "3" if _hits_bumper(o) else "1"
    if o.kind == "zone":
        body.set("mocap", "true")
        geom.update(contype="0", conaffinity="0")
    elif o.movable:
        ET.SubElement(body, "freejoint", name=name)
        geom.update(contype=contype, conaffinity="1", condim="3", mass=_fmt(o.mass), friction="0.8 0.02 0.01")
    else:
        body.set("mocap", "true")
        geom.update(contype=contype, conaffinity="0", condim="3", friction="0.8 0.02 0.01")
    ET.SubElement(body, "geom", geom)
    return body


def _inline_robot_with_bumper(root: ET.Element, robot_xml: str) -> None:
    """Replace <include file="go1_mjx_feetonly.xml"/> with the robot, plus a bumper on its trunk.
    The trunk sets its inertia explicitly, so the extra geom doesn't change the robot's mass."""
    robot = ET.fromstring(robot_xml)
    trunk = next(b for b in robot.iter("body") if b.get("name") == "trunk")
    ET.SubElement(trunk, "geom", name=BUMPER, type="box", size=_fmt(*BUMPER_HALF_SIZE), pos=_fmt(*BUMPER_POS),
                  group="3", contype="0", conaffinity="2", condim="3", friction="0.6 0.02 0.01")
    for i, el in enumerate(list(root)):
        if el.tag == "include" and el.get("file") == ROBOT_FILE:
            root.remove(el)
            for j, child in enumerate(robot):
                root.insert(i + j, child)
            return
    raise ValueError(f"base scene does not include {ROBOT_FILE}")


def _inline_sensors(root: ET.Element, sensor_xml: str, any_contact: bool) -> None:
    """Replace <include file="sensor_feet.xml"/> with its contents.

    With solid objects in the scene, a foot on a box or ball should count as a
    contact, so the floor-only filter (geom2="floor") is dropped. The sensor
    names stay the same, since the env looks them up by name.
    """
    sensors = ET.fromstring(sensor_xml).find("sensor")
    if any_contact:
        for s in sensors:
            s.attrib.pop("geom2", None)
    for i, el in enumerate(list(root)):
        if el.tag == "include" and el.get("file") == "sensor_feet.xml":
            root.remove(el)
            root.insert(i, sensors)
            return
    root.append(sensors)


def _extend_keyframes(root: ET.Element, movable: list[SceneObject], terrain: str) -> None:
    """Free bodies add 7 qpos each; every keyframe must list them, after the robot's."""
    if not movable:
        return
    extra = []
    for o in movable:
        extra += [*_nominal_pos(o, terrain), 1.0, 0.0, 0.0, 0.0]
    for key in root.iter("key"):
        if key.get("qpos"):
            key.set("qpos", " ".join(key.get("qpos").split()) + " " + _fmt(*extra))
