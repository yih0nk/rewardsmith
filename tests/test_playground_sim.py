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


def test_explicit_options_override_environment_in_parallel_jobs(monkeypatch, tmp_path):
    from rewardsmith.sim import parallel
    monkeypatch.setenv("REWARDSMITH_RANDOMIZE", "1")
    monkeypatch.setenv("REWARDSMITH_PUSH", "0")
    monkeypatch.setenv("REWARDSMITH_ROBUSTNESS", "1")
    seen = {}
    monkeypatch.setattr(parallel, "run_in_processes", lambda jobs, **kwargs: seen.update(kwargs) or [])
    sim = PlaygroundSimulator(preset="smoke", randomize=False, push=True, robustness=False)
    sim.run_many([(WALK_BACKWARDS, tmp_path)])
    assert seen["sim_options"] == {"randomize": False, "push": True, "robustness": False}


def test_gates_run_in_warm_processes_and_fall_back_when_one_dies(monkeypatch):
    from rewardsmith.sim import gate, parallel

    in_worker = []
    monkeypatch.setattr(gate, "check_reward", lambda source, impl=None, scene=None: in_worker.append(source))
    sim = PlaygroundSimulator(preset="smoke")
    sim._pool = parallel.WarmPool(1, fake=True)
    try:
        assert "SyntaxError" in sim.gate("def reward(:")  # answered by the warm (fake) child
        assert sim.gate(WALK_BACKWARDS) is None
        assert in_worker == []
        taken = sim._pool.take(WALK_BACKWARDS)
        assert taken.gated == WALK_BACKWARDS
        taken.kill()
    finally:
        sim.close_warm_pool()

    sim._pool = parallel.WarmPool(1, fake=True)
    try:
        sim._pool.procs[0].proc.kill()
        assert sim.gate(WALK_BACKWARDS) is None  # the dead child is dropped; the worker gates it
        assert in_worker == [WALK_BACKWARDS]
    finally:
        sim.close_warm_pool()
