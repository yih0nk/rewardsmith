import pytest
import json
from pathlib import Path

from rewardsmith.sim import run_one
from rewardsmith.sim.base import RunResult
from rewardsmith.sim.parallel import run_in_processes

SOURCE = "def reward(state, action, prev_action):\n    return {'task': 1.0}\n"


def test_jobs_run_in_child_processes_and_keep_order(tmp_path):
    jobs = [(SOURCE + f"# {i}\n", tmp_path / f"c{i}") for i in range(3)]

    results = run_in_processes(jobs, max_parallel=2, fake=True)

    assert len(results) == 3
    assert all(isinstance(r, RunResult) for r in results)
    # each child trained its own source: the fake's score depends on the source text
    qualities = [r.metrics["fake_quality"] for r in results]
    assert len(set(qualities)) > 1
    for _, out_dir in jobs:
        assert (out_dir / "reward.py").exists()
        assert (out_dir / "train.log").exists()


def test_a_crashed_child_becomes_an_error_string(tmp_path):
    out_dir = tmp_path / "bad"
    assert run_one.main([str(tmp_path / "missing.py"), str(out_dir), "--fake"]) == 1
    data = json.loads((out_dir / run_one.RESULT_FILE).read_text())
    assert data["error"].startswith("FileNotFoundError")


def test_no_jobs_starts_nothing():
    assert run_in_processes([], fake=True) == []


def test_warm_processes_take_jobs_and_spares_are_stopped(tmp_path):
    from rewardsmith.sim.parallel import WarmPool

    pool = WarmPool(3, fake=True)
    warm = list(pool.procs)
    jobs = [(SOURCE + f"# {i}\n", tmp_path / f"c{i}") for i in range(2)]

    results = run_in_processes(jobs, max_parallel=2, fake=True, pool=pool)
    pool.close()

    assert all(isinstance(r, RunResult) for r in results)
    assert all((d / "train.log").exists() for _, d in jobs)
    # the two jobs ran in warm processes, which exited cleanly; the spare was killed
    assert [wp.proc.returncode for wp in warm[:2]] == [0, 0]
    assert warm[2].proc.returncode not in (None, 0)


def test_a_warm_process_without_a_job_waits_then_exits_when_its_parent_does(tmp_path):
    import subprocess
    import sys
    import time

    parent = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(2)"])
    child = subprocess.Popen([
        sys.executable, "-m", "rewardsmith.sim.run_one", "--serve", str(tmp_path / "job.json"), "--fake",
        "--parent-pid", str(parent.pid),
    ])
    time.sleep(1)
    assert child.poll() is None  # still waiting for a job
    parent.wait()
    assert child.wait(timeout=10) == 0


def test_a_culled_job_is_killed_and_keeps_its_last_eval(tmp_path, monkeypatch):
    import subprocess
    import sys
    from types import SimpleNamespace

    from rewardsmith.sim import parallel
    from rewardsmith.sim.base import Stopped
    from rewardsmith.sim.progress import ProgressWriter

    procs = []

    def slow_start(source, out_dir, *args):
        parallel._prepare(source, out_dir)
        ProgressWriter(out_dir, 100).point(SimpleNamespace(steps=50, wall_s=1.0, episode_reward=2.0, terms={"a": 1.0}))
        procs.append(subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"]))
        return procs[-1]

    monkeypatch.setattr(parallel, "_start", slow_start)
    jobs = [(SOURCE, tmp_path / "c0")]
    results = run_in_processes(jobs, cull=lambda running: {0: "Stopped: test"} if running else {})

    assert isinstance(results[0], Stopped)
    assert results[0].reason == "Stopped: test"
    assert results[0].metrics == {"steps": 50, "episode_reward": 2.0, "term_a": 1.0}
    assert procs[0].poll() is not None  # the sleeper was killed


def test_behavior_metrics_reach_cold_and_warm_children(tmp_path):
    from rewardsmith.sim import parallel

    proc = parallel._start(SOURCE, tmp_path / "cold", "smoke", None, True, parallel._child_env(1), behavior_metrics=True)
    proc.wait()
    assert "--behavior-metrics" in proc.args

    pool = parallel.WarmPool(1, fake=True)
    warm = pool.take()
    sent = []
    real_send = parallel.send
    parallel.send = lambda path, payload: (sent.append(payload), real_send(path, payload))
    try:
        parallel._hand_over(warm, SOURCE, tmp_path / "warm", "smoke", None, behavior_metrics=True)
    finally:
        parallel.send = real_send
    assert warm.proc.wait(timeout=60) == 0
    assert sent[0]["kind"] == "train" and sent[0]["behavior_metrics"] is True


@pytest.mark.parametrize("warm", [False, True])
def test_resolved_simulator_options_reach_child_execution(tmp_path, monkeypatch, warm):
    from rewardsmith.sim import parallel
    options = {"randomize": False, "push": True, "robustness": False}
    reward = tmp_path / "reward.py"
    reward.write_text(SOURCE)
    out_dir = tmp_path / "out"
    seen = []

    class Simulator:
        def run(self, source, directory, scene=None):
            return RunResult(metrics={"ok": 1.0})

    def simulator(fake, preset, impl, behavior_metrics=False, sim_options=None):
        seen.append(sim_options)
        return Simulator()

    monkeypatch.setattr(run_one, "_simulator", simulator)
    if warm:
        job = tmp_path / "job.json"
        parallel._hand_over(parallel.WarmProcess(None, job), SOURCE, out_dir, "smoke", None, sim_options=options)
        monkeypatch.setattr(run_one, "_warm_up", lambda fake: None)
        monkeypatch.setattr(run_one, "_redirect_output", lambda path: None)
        assert run_one.main(["--serve", str(job), "--fake"]) == 0
    else:
        commands = []
        monkeypatch.setattr(parallel.subprocess, "Popen", lambda cmd, **kwargs: commands.append(cmd))
        parallel._start(SOURCE, out_dir, "smoke", None, True, {}, sim_options=options)
        assert run_one.main(commands[0][3:]) == 0
    assert seen == [options]


def test_parent_liveness_check_does_not_terminate_a_live_process():
    import subprocess
    import sys

    parent = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        assert run_one._alive(parent.pid)
        assert parent.poll() is None
    finally:
        parent.terminate()
        parent.wait(timeout=10)
    assert not run_one._alive(parent.pid)


@pytest.mark.parametrize("warm", [False, True])
def test_scene_evaluation_plan_survives_training_process_handoff(tmp_path, warm):
    from rewardsmith.agents.fake import FakeScener
    from rewardsmith.sim import parallel
    from rewardsmith.sim.scene import parse

    scene = parse(FakeScener().make("push the box into the zone"))
    spec = [{"metric": "push:box:zone", "op": ">=", "value": 0.8}]
    pool = parallel.WarmPool(1, fake=True) if warm else None
    try:
        [result] = run_in_processes([(SOURCE, tmp_path / "job")], fake=True, max_parallel=1,
                                   scene=scene, pool=pool, timeout_s=20,
                                   sim_options={"evaluation_spec": spec})
    finally:
        if pool:
            pool.close()
    assert isinstance(result, RunResult)
    assert isinstance(result.metrics["push:box:zone"], float)


def test_a_warm_process_gates_until_a_source_passes_then_trains_it(tmp_path):
    from rewardsmith.sim.parallel import WarmPool

    pool = WarmPool(2, fake=True)
    try:
        wp = pool.reserve()
        assert pool.reserve() is not wp  # a reserved process is lent to one gate at a time
        pool.release(pool.procs[1])
        assert "SyntaxError" in wp.gate("def reward(:")  # the writer's message, from the child
        assert wp.gated is None
        assert wp.gate(SOURCE) is None  # the fix passes in the same process
        pool.release(wp)
        assert pool.reserve() is pool.procs[1]  # a process holding a passed source isn't lent again
        pool.release(pool.procs[1])

        other = SOURCE + "# other\n"
        jobs = [(other, tmp_path / "c0"), (SOURCE, tmp_path / "c1")]
        results = run_in_processes(jobs, max_parallel=2, fake=True, pool=pool)
    finally:
        pool.close()

    assert all(isinstance(r, RunResult) for r in results)
    assert wp.proc.returncode == 0
    assert (tmp_path / "c1" / "train.log").exists()  # which process got which job: see the next test


def test_take_prefers_the_process_that_gated_the_source(tmp_path):
    from rewardsmith.sim.parallel import WarmPool

    pool = WarmPool(3, fake=True)
    try:
        first, second = pool.reserve(), pool.reserve()
        assert second.gate(SOURCE) is None
        pool.release(first)
        pool.release(second)
        assert pool.take(SOURCE) is second
        assert pool.take(SOURCE) is first  # no other holder: any free one
    finally:
        pool.close()


def test_a_dead_warm_process_raises_instead_of_answering(tmp_path):
    from rewardsmith.sim.parallel import WarmPool, WarmProcessError

    pool = WarmPool(1, fake=True)
    try:
        wp = pool.reserve()
        wp.proc.kill()
        wp.proc.wait()
        with pytest.raises(WarmProcessError):
            wp.gate(SOURCE)
        pool.release(wp)
        assert pool.procs == [] and pool.take() is None
    finally:
        pool.close()


@pytest.mark.parametrize("warm", [False, True])
def test_a_jobs_checkpoint_and_steps_reach_the_child(tmp_path, monkeypatch, warm):
    from rewardsmith.sim import parallel
    out_dir = tmp_path / "out"
    start = {"init_params": str(tmp_path / "params_5000.pkl"), "steps": 2_000_000}
    seen = []

    class Simulator:
        def run(self, source, directory, scene=None, **kwargs):
            seen.append(kwargs)
            return RunResult(metrics={"ok": 1.0})

    monkeypatch.setattr(run_one, "_simulator", lambda *args, **kwargs: Simulator())
    if warm:
        job = tmp_path / "job.json"
        parallel._hand_over(parallel.WarmProcess(None, job), SOURCE, out_dir, "smoke", None, start=start)
        monkeypatch.setattr(run_one, "_warm_up", lambda fake: None)
        monkeypatch.setattr(run_one, "_redirect_output", lambda path: None)
        assert run_one.main(["--serve", str(job), "--fake"]) == 0
    else:
        commands = []
        monkeypatch.setattr(parallel.subprocess, "Popen", lambda cmd, **kwargs: commands.append(cmd))
        parallel._start(SOURCE, out_dir, "smoke", None, True, {}, start=start)
        assert run_one.main(commands[0][3:]) == 0
    assert seen == [{"init_params": Path(start["init_params"]), "steps": 2_000_000}]


def test_jobs_with_options_run_end_to_end(tmp_path):
    jobs = [(SOURCE, tmp_path / "a"), (SOURCE, tmp_path / "b", {"init_params": tmp_path / "p.pkl", "steps": 7})]
    results = run_in_processes(jobs, max_parallel=2, fake=True)
    assert all(isinstance(r, RunResult) for r in results)
    assert json.loads((tmp_path / "a" / "start.json").read_text()) == {}
    assert json.loads((tmp_path / "b" / "start.json").read_text()) == {"init_params": str(tmp_path / "p.pkl"), "steps": "7"}
    with pytest.raises(ValueError, match="unknown job options"):
        run_in_processes([(SOURCE, tmp_path / "c", {"init": 1})], fake=True)
