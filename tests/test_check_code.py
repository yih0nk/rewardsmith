"""Claude-written success checks: the gate, the clause rules and the evaluation path."""

import pytest

pytest.importorskip("jax")

import numpy as np  # noqa: E402

from rewardsmith.agents import checker  # noqa: E402
from rewardsmith.agents.checker import SpecError  # noqa: E402
from rewardsmith.sim import check_code, scene  # noqa: E402
from rewardsmith.sim.check_code import CheckCodeError  # noqa: E402
from rewardsmith.sim.evaluation import check_trajectories  # noqa: E402

STAIRS = scene.parse({"structures": [{"name": "stairs", "kind": "stairs", "color": "gray", "distance": [1.0, 1.5]}]})
CLIMBED = '''import jax.numpy as jnp

def check(traj):
    v = traj["valid"] > 0
    top = jnp.max(jnp.where(v, traj["base_ground_z"], 0.0))
    return {"climbed": top > 0.2, "top": top}
'''
UPRIGHT = '''import jax.numpy as jnp

def check(traj):
    return {"upright": jnp.all(jnp.where(traj["valid"] > 0, traj["up_z"] > 0.5, True))}
'''


def test_gate_reports_result_types_and_the_standing_robot():
    report = check_code.gate(CLIMBED, STAIRS)
    assert report.keys == {"climbed": "bool", "top": "float"}
    assert report.standing == {"climbed": 0.0, "top": 0.0}


@pytest.mark.parametrize("source, message", [
    ("import os\ndef check(t):\n    return {}", "only jax"),
    ("from jax import numpy\ndef check(t):\n    return {}", "from-imports"),
    ("def check(t):\n    return {'x': t.__class__}", "underscore"),
    ("def check(t):\n    return {'x': open('/etc/passwd')}", "'open' is not allowed"),
    ("def check(t):\n    while True:\n        pass", "while loops"),
    ("def check(t):\n    return {'x': t['nope'][0] > 0}", "doesn't exist"),
    ("def check(t):\n    if t['up_z'][0] > 0:\n        return {'a': 1.0}\n    return {'a': 0.0}", "jnp.where"),
    ("def check(t):\n    return {'x': t['up_z']}", "one scalar per episode"),
    ("def check(t):\n    return {}", "non-empty dict"),
    ("def checker(t):\n    return {}", "must define check"),
])
def test_gate_rejects_unsafe_or_broken_code(source, message):
    with pytest.raises(CheckCodeError, match=message):
        check_code.gate(source, STAIRS)


def test_code_clauses_validate_against_the_codes_results():
    clauses = [{"metric": "check:climbed", "op": ">=", "value": 0.8}, {"metric": "scene_fell", "op": "<=", "value": 0.1}]
    check = checker.validate_check(clauses, CLIMBED, "climb the stairs", STAIRS)
    assert check.code == CLIMBED and [c["metric"] for c in check.clauses] == ["check:climbed", "scene_fell"]
    with pytest.raises(SpecError, match="names no result"):
        checker.validate_check([{"metric": "check:flew", "op": ">=", "value": 0.8}], CLIMBED, "climb the stairs", STAIRS)
    with pytest.raises(SpecError, match="never used"):
        checker.validate_check([{"metric": "upright", "op": ">", "value": 0.8}], CLIMBED, "climb the stairs", STAIRS)


def test_a_check_that_holds_for_a_standing_robot_is_rejected():
    clauses = [{"metric": "check:upright", "op": ">=", "value": 0.8}, {"metric": "scene_fell", "op": "<=", "value": 0.1}]
    with pytest.raises(SpecError, match="just stands where it starts"):
        checker.validate_check(clauses, UPRIGHT, "climb the stairs", STAIRS)
    # unless standing still is the task
    checker.validate_check(clauses, UPRIGHT, "stand still for 3 seconds", STAIRS)


def test_falls_and_uprightness_alone_never_count_as_the_task():
    with pytest.raises(SpecError, match="at least one clause on the task itself"):
        checker.validate_for_scene([{"metric": "fell", "op": "<", "value": 0.5}], "do a little dance", None)


def test_trajectories_mask_after_the_first_fall_and_aggregate():
    reset = {"up_z": np.ones((3,))}
    steps = {"up_z": np.ones((3, 4))}
    done = np.array([[0, 0, 1, 0], [0, 0, 0, 0], [1, 0, 0, 0]], dtype=bool)
    traj = check_trajectories(reset, steps, done, slice(0, 3), 0.02)
    assert traj["up_z"].shape == (3, 5)
    assert traj["valid"].tolist() == [[1, 1, 1, 1, 0], [1, 1, 1, 1, 1], [1, 1, 0, 0, 0]]
    src = "import jax.numpy as jnp\ndef check(t):\n    return {'long': jnp.sum(t['valid']) > 3, 'n': jnp.sum(t['valid'])}\n"
    assert check_code.evaluate(src, traj) == {"check:long": pytest.approx(2 / 3), "check:n": pytest.approx(11 / 3)}


def _gait(contacts, steps=100):
    """An episode whose feet repeat the given contact patterns (rows of 4, FR FL RR RL)."""
    pattern = np.array(contacts, dtype=np.float32)
    feet = np.tile(pattern, (steps // len(pattern) + 1, 1))[:steps]
    return {"foot_contact": feet[None], "valid": np.ones((1, steps), np.float32)}


HOP = '''import jax.numpy as jnp

def check(traj):
    v = traj["valid"] > 0
    flight = jnp.all(traj["foot_contact"] < 0.5, axis=-1) & v
    takeoff = flight[1:] & ~flight[:-1]
    return {"flight_frac": jnp.sum(flight) / jnp.maximum(jnp.sum(v), 1), "flights": jnp.sum(takeoff)}
'''


def test_a_hop_needs_all_four_feet_off_at_the_same_step():
    # Live request 5b84e1b9 "hop with all four feet leaving the ground at the same time" got front feet off
    # >= 0.25, rear feet off >= 0.25 and mean feet off >= 1.5: separate airtimes, which a bound passes.
    bound = _gait([[0, 0, 1, 1], [1, 1, 0, 0]])  # front pair up, then rear pair up: never all four
    pronk = _gait([[1, 1, 1, 1], [0, 0, 0, 0]])  # all four down, then all four up
    front_off = (bound["foot_contact"][0, :, :2].max(axis=-1) == 0).mean()
    rear_off = (bound["foot_contact"][0, :, 2:].max(axis=-1) == 0).mean()
    feet_off_mean = (1 - bound["foot_contact"][0]).sum(axis=-1).mean()
    assert front_off >= 0.25 and rear_off >= 0.25 and feet_off_mean >= 1.5  # the vocabulary check passes the bound
    assert check_code.evaluate(HOP, bound)["check:flight_frac"] == 0.0
    hop = check_code.evaluate(HOP, pronk)
    assert hop["check:flight_frac"] == pytest.approx(0.5) and hop["check:flights"] == pytest.approx(50)
