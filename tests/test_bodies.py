import pytest

from rewardsmith.sim import go1_spec
from rewardsmith.sim.bodies import BODIES, DEFAULT_BODY, Body, get_body, register


def test_the_default_body_is_the_go1():
    assert DEFAULT_BODY == "go1"
    assert get_body() is get_body("go1") is BODIES["go1"]
    assert isinstance(get_body(), Body)


def test_unknown_body_is_refused():
    with pytest.raises(ValueError, match="unknown body"):
        get_body("hexapod")
    with pytest.raises(ValueError, match="twice"):
        register(get_body("go1"))


def test_go1_body_writes_the_same_notes():
    body = get_body("go1")
    assert body.describe() == go1_spec.describe()
    assert body.describe(mode="steer", stock_profile="gait") == go1_spec.describe(mode="steer", stock_profile="gait")
    assert body.metric_names() == go1_spec.POSE_METRICS
    assert body.capabilities.can_walk and body.capabilities.legs == 4 and body.skill == "walker"


def test_go1_body_matches_the_go1_env():
    pytest.importorskip("mujoco_playground")
    from rewardsmith.sim import go1

    body = get_body("go1")
    assert (body.nq, body.nv, body.base_env) == (go1.ROBOT_NQ, go1.ROBOT_NV, go1.BASE_ENV)
    assert body.push_overrides == go1.PUSH_OVERRIDES


def test_steer_mode_drives_the_skill_checkpoint(monkeypatch):
    seen = {}
    pytest.importorskip("mujoco_playground")
    from rewardsmith.sim import go1

    monkeypatch.setattr(go1, "make_env", lambda source, **kw: seen.update(kw))
    get_body("go1").make_env("src", mode="steer")
    assert seen["walker"] == get_body("go1").skill_checkpoint
    get_body("go1").make_env("src", mode="steer", walker="other.pkl")
    assert seen["walker"] == "other.pkl"


def test_render_cameras_come_from_the_body():
    pytest.importorskip("mujoco")
    from rewardsmith.sim import render

    assert set(render.cameras_for("go1")) == {"chase", "side"}
    assert render.CAMERAS["chase"].distance == 1.35


def test_simulator_takes_a_body(monkeypatch):
    pytest.importorskip("mujoco_playground")
    from rewardsmith.sim.playground import PlaygroundSimulator

    assert PlaygroundSimulator(preset="smoke").body == "go1"
    monkeypatch.setenv("REWARDSMITH_BODY", "go1")
    sim = PlaygroundSimulator(preset="smoke", walker="")
    assert sim.sim_options()["body"] == "go1" and sim.mode == "off"
    with pytest.raises(ValueError, match="unknown body"):
        PlaygroundSimulator(preset="smoke", body="hexapod")
    with pytest.raises(ValueError, match="unknown body"):
        sim.set_body("hexapod")
    sim.body = "go1"
    assert sim.body == "go1"
    with pytest.raises(ValueError, match="unknown body"):
        sim.body = "hexapod"


def test_the_registry_imports_without_jax():
    import subprocess
    import sys

    code = ("import sys; from rewardsmith.sim.bodies import get_body; get_body().describe(); "
            "print(any(m in sys.modules for m in ('jax', 'mujoco', 'mujoco_playground')))")
    assert subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.strip() == "False"
