from pathlib import Path

import pytest

pytest.importorskip("mujoco_playground")

from rewardsmith.sim.go1_spec import describe  # noqa: E402
from rewardsmith.sim.playground import PlaygroundSimulator  # noqa: E402

WALK_BACKWARDS = (Path(__file__).parent.parent / "examples" / "rewards" / "walk_backwards.py").read_text()


def test_simulator_uses_the_go1_contract_and_gate():
    sim = PlaygroundSimulator(preset="smoke")
    assert sim.describe() == describe()
    assert sim.gate(WALK_BACKWARDS) is None
    assert "SyntaxError" in sim.gate("def reward(:")


def test_unsupported_env_is_refused():
    with pytest.raises(ValueError):
        PlaygroundSimulator("Go1Handstand")
