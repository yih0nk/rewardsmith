"""Scene task outcomes from sampled trajectories, independent of JAX/MuJoCo.

All metrics are episode success rates. Composite metrics join predicates in the
same episode; sequences additionally require each stage after the previous one.
"""

import math
from dataclasses import dataclass
from typing import Any

from .scene import EMPTY, SceneObject, SceneSpec


@dataclass(frozen=True)
class Event:
    description: str
    hold_s: float = 0.0


def catalog(scene: SceneSpec = EMPTY) -> dict[str, Event]:
    events = {
        "stand_still": Event("upright, speed <= 0.1 m/s and turning <= 0.2 rad/s for 1 second", 1.0),
        "move": Event("displacement of at least 0.5 m from the start"),
        "forward": Event("at least 0.5 m displacement along the initial forward heading"),
        "backward": Event("at least 0.5 m displacement against the initial forward heading"),
        "spin": Event("complete a full turn while staying within 0.5 m of the start"),
    }
    for obj in scene.objects:
        events[f"reach:{obj.name}"] = Event(f"robot reaches {obj.name} before termination")
        if obj.kind != "zone":
            events[f"avoid:{obj.name}"] = Event(
                f"robot and feet clear {obj.name} throughout the episode, with at least 0.5 m progress")
        if obj.kind == "box" and not obj.movable:
            events[f"climb:{obj.name}"] = Event(f"upright robot supported atop {obj.name} for 0.5 seconds", 0.5)
            events[f"stand_on:{obj.name}"] = Event(f"stand still supported atop {obj.name} for 1 second", 1.0)
        if obj.movable:
            for zone in scene.objects:
                if zone.kind == "zone":
                    events[f"push:{obj.name}:{zone.name}"] = Event(
                        f"move {obj.name} at least 0.1 m after robot proximity, from outside to fully inside "
                        f"{zone.name}, remain inside at episode end, and settle there for 0.5 seconds", 0.5)
    return events


def event(name: str, scene: SceneSpec) -> Event:
    base, separator, duration = name.partition("@")
    definition = catalog(scene).get(base)
    if definition is None:
        raise ValueError(f"Unknown scene event {name!r}")
    if not separator:
        return definition
    try:
        seconds = float(duration)
    except ValueError as exc:
        raise ValueError(f"Invalid event duration {name!r}") from exc
    if not math.isfinite(seconds) or not 0 < seconds <= 10:
        raise ValueError("Event holding duration must be greater than zero and at most 10 seconds")
    return Event(definition.description + f" (hold for {seconds:g} seconds)", seconds)


def parts(metric: str, scene: SceneSpec) -> tuple[str, list[str]]:
    """Validate a primitive, conjunction, or ordered metric. No nested expressions."""
    try:
        event(metric, scene)
        return "event", [metric]
    except ValueError:
        pass
    for prefix, separator in (("all:", "&"), ("sequence:", ">")):
        if metric.startswith(prefix):
            names = metric[len(prefix):].split(separator)
            if 2 <= len(names) <= 6:
                for name in names:
                    event(name, scene)
                return prefix[:-1], names
    raise ValueError(f"Unknown scene outcome {metric!r}; use named scene events, all:A&B, or sequence:A>B")


def requested(spec: list[dict[str, Any]] | None, scene: SceneSpec) -> list[str]:
    return [c["metric"] for c in spec or [] if c["metric"] not in ("target_reached", "target_fell", "target_final_dist")
            and (c["metric"].split("@")[0] in catalog(scene) or c["metric"].startswith(("all:", "sequence:")))]


def _xy(point, origin, yaw=0.0):
    dx, dy = point[0] - origin[0], point[1] - origin[1]
    c, s = math.cos(yaw), math.sin(yaw)
    return c * dx + s * dy, -s * dx + c * dy


def _distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _clearance(point, pose, obj: SceneObject):
    x, y = _xy(point, pose["pos"], pose["yaw"])
    if obj.kind == "box":
        return math.hypot(max(abs(x) - obj.size[0] / 2, 0), max(abs(y) - obj.size[1] / 2, 0))
    return math.hypot(x, y) - obj.size[0]


def _corners(pose, obj):
    rotation = pose.get("rotation")
    if rotation is None:
        c, s = math.cos(pose["yaw"]), math.sin(pose["yaw"])
        rotation = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    return [[pose["pos"][i] + sum(rotation[i][j] * local[j] for j in range(3)) for i in range(3)]
            for x in (-obj.size[0] / 2, obj.size[0] / 2)
            for y in (-obj.size[1] / 2, obj.size[1] / 2)
            for z in (-obj.size[2] / 2, obj.size[2] / 2) for local in [(x, y, z)]]


def _radius(pose, obj):
    return max(_distance(corner, pose["pos"]) for corner in _corners(pose, obj)) if obj.kind == "box" else obj.size[0]


def _inside(pose, obj: SceneObject, zone_pose, zone: SceneObject):
    if obj.kind == "ball":
        horizontal = _distance(pose["pos"], zone_pose["pos"]) + obj.size[0] <= zone.size[0]
    else:
        corners = _corners(pose, obj)
        horizontal = all(_distance(point, zone_pose["pos"]) <= zone.size[0] for point in corners)
    # A flying object above the zone has not been delivered to it. Tipped boxes
    # use all eight transformed corners, not an upright footprint approximation.
    bottom = min(point[2] for point in corners) if obj.kind == "box" else pose["pos"][2] - obj.rest_height
    return horizontal and abs(bottom - zone_pose["pos"][2]) <= 0.1


def _on_box(frame, obj: SceneObject):
    pose = frame["objects"][obj.name]
    x, y = _xy(frame["base"], pose["pos"], pose["yaw"])
    top = pose["pos"][2] + obj.size[2] / 2
    supported = 0
    for foot, contact in zip(frame["feet"], frame["contacts"]):
        fx, fy = _xy(foot, pose["pos"], pose["yaw"])
        supported += bool(contact and abs(fx) <= obj.size[0] / 2 + 0.02 and
                          abs(fy) <= obj.size[1] / 2 + 0.02 and abs(foot[2] - top) <= 0.04)
    return (frame["up"] >= 0.7 and abs(x) <= obj.size[0] / 2 and abs(y) <= obj.size[1] / 2
            and frame["base"][2] >= top + 0.1 and supported >= 2)


def _still(frame):
    return frame["up"] >= 0.7 and frame["speed"] <= 0.1 and abs(frame["yaw_rate"]) <= 0.2


def _finite(value):
    if isinstance(value, dict):
        return all(_finite(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite(v) for v in value)
    return isinstance(value, (int, float, bool)) and math.isfinite(value)


def _signals(name, scene, start, frames, dt):
    objects = {obj.name: obj for obj in scene.objects}
    kind, *args = name.split("@")[0].split(":")
    if kind == "stand_still":
        return [_still(frame) for frame in frames]
    if kind == "move":
        return [_distance(frame["base"], start["base"]) >= 0.5 for frame in frames]
    if kind in ("forward", "backward"):
        direction = 1 if kind == "forward" else -1
        return [direction * _xy(frame["base"], start["base"], start.get("yaw", 0))[0] >= 0.5 for frame in frames]
    if kind == "spin":
        angle, near_start, values = 0.0, True, []
        for frame in frames:
            angle += frame["yaw_rate"] * dt
            near_start = near_start and _distance(frame["base"], start["base"]) <= 0.5
            values.append(abs(angle) >= math.tau and near_start and frame["up"] >= 0.7)
        return values
    obj = objects[args[0]]
    if kind == "reach":
        return [_distance(frame["base"], frame["objects"][obj.name]["pos"]) <= obj.reach_radius for frame in frames]
    if kind in ("climb", "stand_on"):
        return [_on_box(frame, obj) and (kind == "climb" or _still(frame)) for frame in frames]
    if kind == "avoid":
        previous = start
        safe = True
        for frame in [start, *frames]:
            # Circumscribed footprint circles conservatively cover rotated boxes,
            # trunk and limbs. Test swept segments so a fast crossing cannot vanish
            # between control samples. This is geometric clearance, not contact force.
            before = _xy(previous["base"], previous["objects"][obj.name]["pos"])
            after = _xy(frame["base"], frame["objects"][obj.name]["pos"])
            radius = max([0.35, *[_distance(f, state["base"]) + 0.025
                                  for state in (previous, frame) for f in state["feet"]]])
            dx, dy = after[0] - before[0], after[1] - before[1]
            length = dx * dx + dy * dy
            fraction = max(0.0, min(1.0, -(before[0] * dx + before[1] * dy) / length)) if length else 0.0
            distance = math.hypot(before[0] + fraction * dx, before[1] + fraction * dy)
            obstacle_radius = max(_radius(previous["objects"][obj.name], obj), _radius(frame["objects"][obj.name], obj))
            safe = safe and distance > obstacle_radius + radius
            previous = frame
        progress = max((_distance(frame["base"], start["base"]) for frame in frames), default=0) >= 0.5
        return [safe and progress] * len(frames)
    zone = objects[args[1]]
    initial = start["objects"][obj.name]
    began_outside = not _inside(initial, obj, start["objects"][zone.name], zone)
    ends_inside = _inside(frames[-1]["objects"][obj.name], obj, frames[-1]["objects"][zone.name], zone)
    pushed_distance = 0.0
    values = []
    previous_pose = initial
    previous_frame = start
    for frame in [start, *frames]:
        pose = frame["objects"][obj.name]
        near = _clearance(previous_frame["base"], previous_pose, obj) <= 0.4 or any(
            _clearance(foot, previous_pose, obj) <= 0.06 for foot in previous_frame["feet"])
        if near:
            pushed_distance += _distance(previous_pose["pos"], pose["pos"])
        moved = _distance(initial["pos"], pose["pos"]) >= 0.1
        inside = _inside(pose, obj, frame["objects"][zone.name], zone)
        values.append(began_outside and ends_inside and pushed_distance >= 0.1 and moved and inside
                      and _distance(previous_pose["pos"], pose["pos"]) / dt <= 0.1)
        previous_pose = pose
        previous_frame = frame
    return values[1:]


def _complete(values, hold_s, dt, after=-1):
    needed = max(1, math.ceil(hold_s / dt - 1e-9))
    streak = 0
    for index in range(after + 1, len(values)):
        streak = streak + 1 if values[index] else 0
        if streak >= needed:
            return index
    return None


def evaluate(scene: SceneSpec, episodes: list[dict], dt: float, metrics: list[str]) -> tuple[dict[str, float], list[dict]]:
    """Return rates and per-episode evidence. Ignore every sample after first done.

    frames hold base/feet poses, contact flags, uprightness, speed, yaw rate,
    object poses, and done. The initial reset sample is separate from frames.
    """
    if not math.isfinite(dt) or dt <= 0 or not episodes:
        raise ValueError("Scene evaluation requires episodes and a positive control timestep")
    plans = {metric: parts(metric, scene) for metric in metrics}
    rows = []
    for episode in episodes:
        frames = []
        for frame in episode["frames"]:
            frames.append(frame)
            if frame["done"]:
                break
        valid = bool(frames) and _finite(episode["start"]) and _finite(frames)
        signals = {}
        row = {"valid": valid, "fell": bool(not valid or frames[-1]["done"]), "outcomes": {}}
        for metric, (mode, names) in plans.items():
            completions = []
            if valid:
                for name in names:
                    if name not in signals:
                        signals[name] = _signals(name, scene, episode["start"], frames, dt)
                    after = completions[-1] if mode == "sequence" and completions else -1
                    at = _complete(signals[name], event(name, scene).hold_s, dt, after)
                    if at is None:
                        break
                    completions.append(at)
            passed = len(completions) == len(names)
            row["outcomes"][metric] = {"pass": passed, "completed_at_s": [(i + 1) * dt for i in completions]}
        rows.append(row)
    summary = {metric: sum(row["outcomes"][metric]["pass"] for row in rows) / len(rows) for metric in metrics}
    summary["scene_fell"] = sum(row["fell"] for row in rows) / len(rows)
    return summary, rows
