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

    def slow_check(source, impl=None, scene=None, stock=None):
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
    assert {k: seen["sim_options"][k] for k in ("randomize", "push", "robustness", "evaluation_spec")} == {
        "randomize": False, "push": True, "robustness": False, "evaluation_spec": None}


def test_children_get_the_warm_start_settings_resolved(monkeypatch, tmp_path):
    from rewardsmith.sim import parallel
    walker = tmp_path / "params.pkl"
    walker.write_bytes(b"")
    monkeypatch.setenv("REWARDSMITH_WALKER", str(walker))
    monkeypatch.delenv("REWARDSMITH_STOCK", raising=False)
    monkeypatch.delenv("REWARDSMITH_STEPS", raising=False)
    seen = {}
    monkeypatch.setattr(parallel, "run_in_processes", lambda jobs, **kwargs: seen.update(kwargs, jobs=jobs) or [])
    sim = PlaygroundSimulator(preset="smoke")
    sim.stock_profile = "pose"
    sim.run_many([(WALK_BACKWARDS, tmp_path / "a"), (WALK_BACKWARDS, tmp_path / "b", {"steps": 5})])
    opts = seen["sim_options"]
    assert opts["walker"] == str(walker) and opts["stock_profile"] == "pose" and opts["steps"] is None
    assert seen["jobs"][1][2] == {"steps": 5}
    assert PlaygroundSimulator(**opts).sim_options() == opts  # a child trains exactly as the parent would


def test_no_walker_means_no_warm_start_and_no_stock_terms(monkeypatch, tmp_path):
    monkeypatch.setenv("REWARDSMITH_WALKER", str(tmp_path / "missing.pkl"))
    monkeypatch.delenv("REWARDSMITH_STOCK", raising=False)
    sim = PlaygroundSimulator(preset="smoke")
    assert sim.walker == "" and sim.stock_profile == "off"
    monkeypatch.setenv("REWARDSMITH_WARM_START", "0")
    assert PlaygroundSimulator(preset="smoke").walker == ""
