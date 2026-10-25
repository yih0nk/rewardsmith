"""Claude-built scenes: capsules, tilt, elevation, launched objects and structures."""

import json

import pytest

from rewardsmith.sim import scene
from rewardsmith.sim.scene import SceneError

POLE = {"name": "pole1", "kind": "capsule", "color": "red", "size": [0.05, 1.0], "distance": 2.0, "bearing": 0}
STAIRS = {"name": "stairs", "kind": "stairs", "color": "gray", "distance": [1.0, 1.5], "bearing": [-10, 10]}


def test_capsules_tilt_and_elevation_parse_with_geometry():
    bar = scene.parse({"objects": [{**POLE, "name": "bar", "tilt": [90, 0], "elevation": 0.1}]}).objects[0]
    assert bar.rest_height == pytest.approx(0.1 + 0.05)  # lying down: its radius above its elevation
    assert bar.top_height == pytest.approx(0.2)
    pole = scene.parse({"objects": [POLE]}).objects[0]
    assert pole.rest_height == pytest.approx(0.05 + 0.5) and pole.footprint_radius == pytest.approx(0.05)


def test_structures_round_trip_and_default_their_params():
    s = scene.parse({"structures": [STAIRS, {"name": "ramp", "kind": "ramp", "color": "orange", "distance": 1.0,
                                             "bearing": 120, "params": {"angle": 12}}]})
    assert scene.parse(s.to_json()) == s
    stairs, ramp = s.structures
    assert stairs.p("steps") == 4 and len(stairs.pieces) == 4
    assert stairs.top_height == pytest.approx(4 * 0.06)
    assert ramp.top_height == pytest.approx(2.0 * 0.21256, abs=1e-3)
    assert s.target == "stairs" and s.target_structure is stairs and s.target_object is None
    assert s.has_ground and not s.is_empty


def test_legacy_specs_are_unchanged():
    legacy = {"name": "ball", "kind": "ball", "color": "red", "size": [0.15], "distance": [1.5, 3.0]}
    s = scene.parse({"objects": [legacy]})
    o = s.objects[0]
    assert o.relevant and o.tilt == (0.0, 0.0) and o.speed == (0.0, 0.0) and not s.structures
    assert o.rest_height == 0.15 and json.loads(s.to_json())["structures"] == []


@pytest.mark.parametrize("data, message", [
    ({"objects": [{**POLE, "speed": [1, 2]}]}, "only movable objects can be launched"),
    ({"objects": [{**POLE, "movable": True, "elevation": 0.2}]}, "only fixed solid objects can float"),
    ({"objects": [{**POLE, "kind": "ball", "size": [0.1], "tilt": [10, 0]}]}, "can't be tilted"),
    ({"structures": [{**STAIRS, "params": {"rise": 0.5}}]}, "rise must be between"),
    ({"structures": [{**STAIRS, "params": {"steps": 2.5}}]}, "steps must be a whole number"),
    ({"structures": [{**STAIRS, "params": {"angle": 3}}]}, "unknown stairs param"),
    ({"structures": [{**STAIRS, "kind": "trampoline"}]}, "kind must be one of"),
    ({"structures": [{**STAIRS, "distance": 0.2}]}, "comes within"),
    ({"structures": [{"name": "f", "kind": "field", "color": "red", "distance": 1,
                      "params": {"count": 12, "size": 0.4, "length": 1, "width": 1}}]}, "obstacles fit"),
    ({"structures": [{**STAIRS, "name": f"s{i}", "params": {"steps": 8}} for i in range(4)]}, "solid pieces"),
    ({"objects": [{**POLE, "name": f"p{i}", "distance": 1.5 + i} for i in range(6)],
      "structures": [STAIRS]}, "relevant objects and structures"),
    ({"objects": [{**POLE, "relevant": False}]}, "must be relevant"),
])
def test_bad_new_specs_are_rejected_with_a_fix(data, message):
    with pytest.raises(SceneError, match=message):
        scene.parse(data)


def test_summary_describes_structures_and_launches():
    s = scene.parse({"objects": [{"name": "ball", "kind": "ball", "color": "red", "size": [0.12], "distance": [4, 5],
                                  "movable": True, "mass": 0.5, "speed": [1.5, 2.5]}], "structures": [STAIRS]})
    text = s.summary()
    assert "launched at 1.5 to 2.5 m/s" in text and "4 steps of 0.06 m rise" in text
