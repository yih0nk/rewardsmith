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


def test_gates_from_threads_take_turns(monkeypatch):
    import threading
    import time
    from concurrent.futures import ThreadPoolExecutor

    from rewardsmith.sim import gate

    inside, peak, lock = [0], [0], threading.Lock()

    def slow_check(source, impl=None, scene=None):
        with lock:
            inside[0] += 1
            peak[0] = max(peak[0], inside[0])
        time.sleep(0.05)
        with lock:
            inside[0] -= 1
        return None

    monkeypatch.setattr(gate, "check_reward", slow_check)
    sim = PlaygroundSimulator(preset="smoke")
    with ThreadPoolExecutor(4) as pool:
        assert list(pool.map(sim.gate, ["a", "b", "c", "d"])) == [None] * 4
    assert peak[0] == 1  # warp's CUDA graph capture breaks if two gates run at once
