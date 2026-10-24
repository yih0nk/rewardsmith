import json

import pytest

from rewardsmith.sim import scene
from rewardsmith.sim.scene import SceneError

BALL = {"name": "ball", "kind": "ball", "color": "red", "size": [0.15], "distance": [1.5, 3.0], "bearing": [-60, 60]}


def spec(*objects, **top):
    return {"terrain": "flat", "objects": list(objects), **top}


def test_empty_and_missing_specs_are_the_plain_floor():
    assert scene.parse(None).is_empty
    assert scene.parse({}).is_empty
    assert scene.parse("{}") == scene.EMPTY
    assert not scene.parse({"terrain": "rough"}).is_empty


def test_ball_spec_parses_with_defaults():
    s = scene.parse(spec(BALL))
    (ball,) = s.objects
    assert s.target == "ball"  # defaults to the first object
    assert ball.size == (0.15,) and ball.distance == (1.5, 3.0) and ball.bearing == (-60.0, 60.0)
    assert ball.yaw == (0.0, 0.0) and ball.movable is False
    assert ball.rgb == scene.COLORS["red"]
    assert s.target_object is ball


def test_numbers_fix_a_range():
    s = scene.parse(spec({**BALL, "distance": 2, "bearing": 30.0}))
    assert s.objects[0].distance == (2.0, 2.0)
    assert s.objects[0].bearing == (30.0, 30.0)


def test_round_trips_through_json():
    s = scene.parse(spec(BALL, {"name": "crate", "kind": "box", "color": "blue", "size": [0.4, 0.4, 0.3],
                                "distance": 2.5, "movable": True, "mass": 2.0}, target="crate"))
    assert scene.parse(s.to_json()) == s
    assert json.loads(s.to_json())["target"] == "crate"


def test_summary_names_every_object():
    s = scene.parse(spec(BALL, {"name": "goal", "kind": "zone", "color": "green", "size": [0.5], "distance": 3}))
    text = s.summary()
    assert "ball: red ball" in text and "goal: green zone" in text and "target object is ball" in text


@pytest.mark.parametrize(
    "data, message",
    [
        ("not json", "not valid JSON"),
        ([1, 2], "must be a JSON object"),
        ({"terrain": "lava"}, "terrain must be one of"),
        ({"floor": "flat"}, "unknown key(s) 'floor'"),
        (spec({**BALL, "kind": "teapot"}), "kind must be one of"),
        (spec({**BALL, "color": "chartreuse"}), "color must be one of"),
        (spec({**BALL, "name": "Ball!"}), "name must be lowercase"),
        (spec({**BALL, "name": "base"}), "reserved"),
        (spec({**BALL, "size": [2.0]}), "radius must be between"),
        (spec({**BALL, "size": [0.1, 0.2]}), "size must be [radius]"),
        (spec({**BALL, "kind": "box", "size": [0.5, 0.5]}), "[length, width, height]"),
        (spec({**BALL, "kind": "box", "size": [1.5, 0.5, 0.5], "movable": True}), "length must be between 0.05 and 1.0"),
        (spec({**BALL, "distance": [3, 1]}), "low above high"),
        (spec({**BALL, "distance": [1, 20]}), "must stay within [0, 8]"),
        (spec({**BALL, "distance": 0.2}), "does not spawn on the robot"),
        (spec({**BALL, "bearing": [0, 270]}), "must stay within [-180, 180]"),
        (spec({**BALL, "distance": float("nan")}), "distance must be a number"),
        (spec({**BALL, "movable": "yes"}), "movable must be true or false"),
        (spec({**BALL, "movable": True, "mass": 500}), "mass must be between"),
        (spec({**BALL, "kind": "zone", "size": [0.5], "movable": True}), "cannot be movable"),
        (spec({**BALL, "wobble": 1}), "unknown key(s) 'wobble'"),
        (spec(BALL, BALL), "used more than once"),
        (spec(*[{**BALL, "name": f"b{i}"} for i in range(7)]), "at most 6 objects"),
        (spec(BALL, target="cube"), "not an object name"),
        ({"target": "ball"}, "the scene has none"),
    ],
)
def test_bad_specs_explain_themselves(data, message):
    with pytest.raises(SceneError, match="scene spec") as info:
        scene.parse(data)
    assert message in str(info.value)


def test_every_problem_is_reported_at_once():
    with pytest.raises(SceneError) as info:
        scene.parse(spec({**BALL, "color": "chartreuse", "size": [9.0]}, terrain="lava"))
    text = str(info.value)
    assert "terrain" in text and "color" in text and "radius" in text


def test_zone_may_sit_under_the_robot():
    s = scene.parse(spec({"name": "circle", "kind": "zone", "color": "green", "size": [1.0], "distance": 0}))
    assert s.objects[0].reach_radius == 1.0


# Compiling needs Playground's Go1 files and MuJoCo.

def _model(s):
    mujoco = pytest.importorskip("mujoco")
    pytest.importorskip("mujoco_playground")
    from mujoco_playground._src.locomotion.go1 import base

    from rewardsmith.sim import scene_xml

    return mujoco.MjModel.from_xml_string(scene_xml.write_scene_xml(s).read_text(), assets=base.get_assets())


FULL = spec(
    BALL,
    {"name": "crate", "kind": "box", "color": "blue", "size": [0.4, 0.4, 0.3], "distance": 2.5, "movable": True, "mass": 2},
    {"name": "goal", "kind": "zone", "color": "green", "size": [0.5], "distance": 3, "bearing": 180},
    {"name": "step", "kind": "box", "color": "gray", "size": [1.0, 1.5, 0.08], "distance": 2, "bearing": -90},
)


@pytest.mark.parametrize("terrain", ["flat", "rough"])
def test_compiled_scene_loads_with_robot_first(terrain):
    m = _model(scene.parse({**FULL, "terrain": terrain}))
    assert m.nq == 19 + 7 and m.nv == 18 + 6  # one free body (crate), after the robot
    assert m.nmocap == 3  # ball, goal and step are fixed and moved by reset
    assert m.jnt_qposadr[m.joint("obj_crate").id] == 19
    assert all(m.key(i).qpos.shape == (m.nq,) for i in range(m.nkey))
    for name in ("obj_ball", "obj_crate", "obj_goal", "obj_step"):
        assert m.geom(name).group == 0  # visible in the rollout video
    assert m.geom("obj_goal").contype == 0 and m.geom("obj_goal").conaffinity == 0
    assert m.camera("overview") is not None


def test_contact_budget_grows_with_solid_objects():
    from rewardsmith.sim import scene_xml

    assert scene_xml.contact_budget(scene.EMPTY) == (4 * 8192, 40)
    naconmax, njmax = scene_xml.contact_budget(scene.parse(FULL))
    assert naconmax > 4 * 8192 and njmax > 40


def test_foot_sensors_count_objects_only_when_there_are_solid_ones():
    from rewardsmith.sim import scene_xml

    sensors = '<mujoco><sensor><contact name="FR_floor_found" geom1="FR" geom2="floor" data="found"/></sensor></mujoco>'
    base = '<mujoco><worldbody/><include file="sensor_feet.xml"/></mujoco>'
    zone_only = scene.parse(spec({"name": "goal", "kind": "zone", "color": "green", "size": [0.5], "distance": 2}))
    assert 'geom2="floor"' in scene_xml.compile_xml(zone_only, base, sensors)
    assert 'geom2="floor"' not in scene_xml.compile_xml(scene.parse(spec(BALL)), base, sensors)
    assert "include" not in scene_xml.compile_xml(scene.parse(spec(BALL)), base, sensors)
