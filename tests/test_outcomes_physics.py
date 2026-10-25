"""Outcome checks against native MuJoCo physics, without a trained policy.

The robot follows an imposed pose trajectory; the crate remains a free body.
These tests validate measurement and contact geometry, not learned task ability.
"""

import math

import pytest

mujoco = pytest.importorskip("mujoco")
np = pytest.importorskip("numpy")
pytest.importorskip("mujoco_playground")

from mujoco_playground._src.locomotion.go1 import base  # noqa: E402

from rewardsmith.sim import outcomes, scene, scene_xml  # noqa: E402


def model_for(spec):
    return mujoco.MjModel.from_xml_string(
        scene_xml.write_scene_xml(spec).read_text(), assets=base.get_assets())


def observe(model, data, spec):
    trunk = model.body("trunk").id
    foot_names = ("FR", "FL", "RR", "RL")
    pairs = [(c.geom1, c.geom2) for c in data.contact[:data.ncon]]
    contacts = [any(model.geom(name).id in pair for pair in pairs) for name in foot_names]
    return {
        "base": data.qpos[:3].tolist(), "yaw": math.atan2(data.xmat[trunk, 3], data.xmat[trunk, 0]),
        "up": float(data.xmat[trunk, 8]), "speed": float(np.linalg.norm(data.qvel[:2])),
        "yaw_rate": float(data.qvel[5]),
        "feet": [data.site_xpos[model.site(name).id].tolist() for name in foot_names],
        "contacts": contacts, "done": False,
        "objects": {obj.name: {
            "pos": data.xpos[model.body("obj_" + obj.name).id].tolist(),
            "yaw": math.atan2(data.xmat[model.body("obj_" + obj.name).id, 3],
                              data.xmat[model.body("obj_" + obj.name).id, 0]),
            "rotation": data.xmat[model.body("obj_" + obj.name).id].reshape(3, 3).tolist(),
        } for obj in spec.objects},
    }


def test_physical_crate_push_and_settling_are_recognized():
    spec = scene.parse({"objects": [
        {"name": "crate", "kind": "box", "size": [0.2, 0.2, 0.2],
         "distance": 0.7, "bearing": 0, "movable": True, "mass": 0.5},
        {"name": "goal", "kind": "zone", "size": [0.5], "distance": 1.3, "bearing": 0},
    ]})
    model = model_for(spec)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    robot_pose = data.qpos[:19].copy()
    mujoco.mj_forward(model, data)
    start = observe(model, data, spec)
    frames, touched = [], False
    bumper, crate = model.geom("obj_bumper").id, model.geom("obj_crate").id
    substeps = round(0.02 / model.opt.timestep)
    for index in range(150):
        for substep in range(substeps):
            # A scripted kinematic robot pushes a dynamically simulated crate.
            data.qpos[:19] = robot_pose
            data.qpos[0] += min((index + (substep + 1) / substeps) * 0.02, 1.0)
            data.qvel[:18] = 0
            data.ctrl[:] = robot_pose[7:19]
            mujoco.mj_step(model, data)
            touched |= any({c.geom1, c.geom2} == {bumper, crate} for c in data.contact[:data.ncon])
        mujoco.mj_forward(model, data)
        frames.append(observe(model, data, spec))
    assert touched, "The positive case must involve a real robot/crate collision"
    assert frames[-1]["objects"]["crate"]["pos"][0] - start["objects"]["crate"]["pos"][0] > 0.1
    rates, _ = outcomes.evaluate(spec, [{"start": start, "frames": frames}], 0.02, ["push:crate:goal"])
    assert rates["push:crate:goal"] == 1


@pytest.mark.parametrize("supported", [True, False])
def test_platform_support_uses_real_foot_contacts(supported):
    spec = scene.parse({"objects": [
        {"name": "step", "kind": "box", "size": [1.0, 1.0, 0.1], "distance": 2, "bearing": 0},
    ]})
    model = model_for(spec)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    data.qpos[0] = 2
    # Lower the initialized feet a fraction into the platform to obtain support contacts.
    mujoco.mj_forward(model, data)
    feet_z = min(data.site_xpos[model.site(name).id, 2] for name in ("FR", "FL", "RR", "RL"))
    data.qpos[2] += 0.1 - feet_z - 0.005 + (0 if supported else 0.2)
    mujoco.mj_forward(model, data)
    frame = observe(model, data, spec)
    assert sum(frame["contacts"]) >= 2 if supported else not any(frame["contacts"])
    pose = data.qpos[:19].copy()
    frames = []
    for _ in range(75):
        for _ in range(round(0.02 / model.opt.timestep)):
            data.qpos[:19] = pose
            data.qvel[:18] = 0
            data.ctrl[:] = pose[7:19]
            mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)
        frames.append(observe(model, data, spec))
    assert data.time == pytest.approx(1.5)
    rates, _ = outcomes.evaluate(spec, [{"start": frame, "frames": frames}], 0.02,
                                 ["sequence:climb:step>stand_on:step"])
    assert rates["sequence:climb:step>stand_on:step"] == int(supported)


@pytest.mark.parametrize("detour", [True, False])
def test_scripted_paths_verify_avoidance_and_reach_order(detour):
    spec = scene.parse({"objects": [
        {"name": "obstacle", "kind": "box", "size": [0.3, 0.3, 0.4], "distance": 1, "bearing": 0},
        {"name": "first", "kind": "zone", "size": [0.25], "distance": math.sqrt(2), "bearing": 45},
        {"name": "second", "kind": "zone", "size": [0.25], "distance": math.sqrt(5),
         "bearing": math.degrees(math.atan2(1, 2))},
    ]})
    model = model_for(spec)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    robot_pose = data.qpos[:19].copy()
    mujoco.mj_forward(model, data)
    start = observe(model, data, spec)
    waypoints = [(0, 0), (0, 1), (1, 1), (2, 1)] if detour else [(0, 0), (2, 0)]
    frames, collided = [], False
    obstacle = model.geom("obj_obstacle").id
    robot_geoms = {model.geom(name).id for name in ("obj_bumper", "FR", "FL", "RR", "RL")}
    for before, after in zip(waypoints, waypoints[1:]):
        for fraction in np.linspace(0, 1, 51)[1:]:
            data.qpos[:19] = robot_pose
            data.qpos[:2] = np.array(before) + fraction * (np.array(after) - before)
            data.qvel[:18] = 0
            data.ctrl[:] = robot_pose[7:19]
            mujoco.mj_step(model, data)
            mujoco.mj_forward(model, data)
            collided |= any(obstacle in (c.geom1, c.geom2) and
                            bool(robot_geoms.intersection((c.geom1, c.geom2)))
                            for c in data.contact[:data.ncon])
            frames.append(observe(model, data, spec))
    assert collided == (not detour)
    ordered = "sequence:reach:first>reach:second"
    reversed_order = "sequence:reach:second>reach:first"
    together = "all:avoid:obstacle&reach:second"
    rates, _ = outcomes.evaluate(spec, [{"start": start, "frames": frames}], model.opt.timestep,
                                 ["avoid:obstacle", ordered, reversed_order, together])
    assert rates["avoid:obstacle"] == int(detour)
    assert rates[ordered] == int(detour)
    assert rates[reversed_order] == 0
    assert rates[together] == int(detour)
