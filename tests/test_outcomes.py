import copy

import pytest

from rewardsmith.sim import outcomes
from rewardsmith.sim.scene import parse


def world(*objects):
    return parse({"objects": list(objects)})


def obj(name, kind="ball", movable=False, size=None):
    return {"name": name, "kind": kind, "movable": movable, "distance": 2,
            "size": size or ({"ball": [0.15], "box": [0.2, 0.2, 0.2], "zone": [0.5]}[kind])}


def frame(x=0.0, y=0.0, z=0.3, objects=None, done=False, speed=0.0, feet=None, contacts=None):
    return {"base": [x, y, z], "yaw": 0.0, "up": 1.0, "speed": speed, "yaw_rate": 0.0,
            "feet": feet or [[x + dx, y + dy, 0.0] for dx, dy in [(0.1, 0.1), (0.1, -0.1), (-0.1, 0.1), (-0.1, -0.1)]],
            "contacts": contacts or [True] * 4, "done": done,
            "objects": {name: {"pos": list(pos), "yaw": 0.0} for name, pos in (objects or {}).items()}}


def measure(scene, start, frames, metric, dt=0.1):
    summary, rows = outcomes.evaluate(scene, [{"start": start, "frames": frames}], dt, [metric])
    return summary[metric], rows[0]


def test_push_requires_delivery_movement_proximity_and_settling():
    scene = world(obj("box", "box", True), obj("goal", "zone"))
    start = frame(x=1.55, objects={"box": (2, 0, 0.1), "goal": (4, 0, 0)})
    delivered = frame(x=3.55, objects={"box": (4, 0, 0.1), "goal": (4, 0, 0)})
    metric = "push:box:goal"
    assert measure(scene, start, [delivered] * 6, metric)[0] == 1
    assert measure(scene, start, [start] * 10, metric)[0] == 0  # just approaching is insufficient
    far = copy.deepcopy(start)
    far["base"] = [0, 0, 0.3]
    far["feet"] = frame()["feet"]
    assert measure(scene, far, [delivered] * 6, metric)[0] == 0  # moving without robot interaction
    assert measure(scene, delivered, [delivered] * 10, metric)[0] == 0  # spawned in the zone
    outside = frame(objects={"box": (5, 0, 0.1), "goal": (4, 0, 0)})
    assert measure(scene, start, [delivered] * 6 + [outside], metric)[0] == 0
    assert measure(scene, start, [delivered] * 2, metric)[0] == 0  # no sustained placement


def test_push_requires_the_whole_object_inside_and_on_the_floor():
    scene = world(obj("box", "box", True, [0.4, 0.4, 0.2]), obj("goal", "zone", size=[0.2]))
    start = frame(x=1.5, objects={"box": (2, 0, 0.1), "goal": (4, 0, 0)})
    centered = frame(objects={"box": (4, 0, 0.1), "goal": (4, 0, 0)})
    assert measure(scene, start, [centered] * 10, "push:box:goal")[0] == 0
    flying = frame(objects={"box": (4, 0, 1.0), "goal": (4, 0, 0)})
    assert measure(scene, start, [flying] * 10, "push:box:goal")[0] == 0


def test_tipped_box_cannot_pass_using_its_smaller_upright_footprint():
    import math
    scene = world(obj("box", "box", True, [0.2, 0.2, 0.6]), obj("goal", "zone", size=[0.25]))
    start = frame(x=1.55, objects={"box": (2, 0, 0.3), "goal": (4, 0, 0)})
    tipped = frame(objects={"box": (4, 0, 0.4 / math.sqrt(2)), "goal": (4, 0, 0)})
    c = 1 / math.sqrt(2)
    tipped["objects"]["box"]["rotation"] = [[c, 0, c], [0, 1, 0], [-c, 0, c]]
    assert measure(scene, start, [tipped] * 10, "push:box:goal")[0] == 0


def test_climb_requires_support_and_stand_requires_a_fresh_still_interval():
    scene = world(obj("step", "box", size=[0.4, 0.4, 0.2]))
    start = frame(objects={"step": (2, 0, 0.1)})
    feet = [[2 + dx, dy, 0.2] for dx, dy in [(0.1, 0.1), (0.1, -0.1), (-0.1, 0.1), (-0.1, -0.1)]]
    atop = frame(x=2, z=0.5, feet=feet, objects={"step": (2, 0, 0.1)})
    assert measure(scene, start, [atop] * 5, "climb:step")[0] == 1
    unsupported = copy.deepcopy(atop)
    unsupported["contacts"] = [False] * 4
    assert measure(scene, start, [unsupported] * 20, "climb:step")[0] == 0
    assert measure(scene, start, [atop] * 14, "sequence:climb:step>stand_on:step")[0] == 0
    assert measure(scene, start, [atop] * 15, "sequence:climb:step>stand_on:step")[0] == 1
    moving = {**atop, "speed": 0.3}
    assert measure(scene, start, [moving] * 20, "stand_on:step")[0] == 0


def test_avoidance_requires_progress_and_detects_a_swept_crossing():
    scene = world(obj("obstacle"))
    start = frame(objects={"obstacle": (2, 0, 0.15)})
    end = frame(x=4, objects={"obstacle": (2, 0, 0.15)})
    assert measure(scene, start, [end], "avoid:obstacle")[0] == 0  # crossed between samples
    assert measure(scene, start, [start] * 20, "avoid:obstacle")[0] == 0  # stayed still
    path = [frame(x=x, y=y, objects={"obstacle": (2, 0, 0.15)}) for x, y in [(0, 1), (4, 1), (4, 0)]]
    assert measure(scene, start, path, "avoid:obstacle")[0] == 1


def test_conjunction_cannot_combine_successes_from_different_episodes():
    scene = world(obj("obstacle"), obj("goal", "zone"))
    positions = {"obstacle": (2, 0, 0.15), "goal": (4, 0, 0)}
    start = frame(objects=positions)
    episodes = [
        {"start": start, "frames": [frame(x=4, objects=positions)]},  # reached, hit obstacle
        {"start": start, "frames": [frame(y=1, objects=positions)]},  # avoided, never reached
    ]
    metrics = ["reach:goal", "avoid:obstacle", "all:reach:goal&avoid:obstacle"]
    result, _ = outcomes.evaluate(scene, episodes, 0.1, metrics)
    assert result["reach:goal"] == 0.5 and result["avoid:obstacle"] == 0.5
    assert result[metrics[-1]] == 0


def test_sequence_checks_order_and_allows_later_revisits():
    scene = world(obj("ball"), obj("goal", "zone"))
    positions = {"ball": (2, 0, 0.15), "goal": (4, 0, 0)}
    start = frame(objects=positions)
    ball, goal = frame(x=2, objects=positions), frame(x=4, objects=positions)
    metric = "sequence:reach:ball>reach:goal"
    assert measure(scene, start, [goal, ball], metric)[0] == 0
    passed, evidence = measure(scene, start, [goal, ball, goal], metric)
    assert passed == 1
    assert evidence["outcomes"][metric]["completed_at_s"] == pytest.approx([0.2, 0.3])


def test_sequence_does_not_reuse_standing_time_from_before_reaching():
    scene = world(obj("ball"))
    start = frame(objects={"ball": (2, 0, 0.15)})
    reach = frame(x=2, speed=0.4, objects={"ball": (2, 0, 0.15)})
    assert measure(scene, start, [start] * 10 + [reach] * 10,
                   "sequence:reach:ball>stand_still")[0] == 0


def test_no_credit_after_falling_or_for_invalid_telemetry():
    scene = world(obj("goal", "zone"))
    start = frame(objects={"goal": (2, 0, 0)})
    fallen = {**start, "done": True}
    reached = frame(x=2, objects={"goal": (2, 0, 0)})
    assert measure(scene, start, [fallen, reached], "reach:goal")[0] == 0
    reached["up"] = float("nan")
    result, rows = outcomes.evaluate(scene, [{"start": start, "frames": [reached]}], 0.1, ["reach:goal"])
    assert result["reach:goal"] == 0 and result["scene_fell"] == 1
    assert rows[0]["valid"] is False


def test_explicit_hold_duration_is_measured_after_the_previous_stage():
    scene = world(obj("ball"))
    start = frame(objects={"ball": (2, 0, 0.15)})
    reached = frame(x=2, objects={"ball": (2, 0, 0.15)})
    metric = "sequence:reach:ball>stand_still@2"
    assert measure(scene, start, [reached] * 20, metric)[0] == 0
    assert measure(scene, start, [reached] * 21, metric)[0] == 1


@pytest.mark.parametrize("duration", ["nan", "inf", "0", "-1", "11", "oops"])
def test_invalid_or_unbounded_holding_durations_are_rejected(duration):
    with pytest.raises(ValueError):
        outcomes.parts("stand_still@" + duration, world())


@pytest.mark.parametrize("metric", ["push:missing:zone", "climb:ball", "sequence:reach:ball", "all:reach:ball&missing"])
def test_invalid_object_references_and_expressions_are_rejected(metric):
    with pytest.raises(ValueError):
        outcomes.parts(metric, world(obj("ball")))
