import json

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
    assert [p.returncode for p, _ in warm[:2]] == [0, 0]
    assert warm[2][0].returncode not in (None, 0)


def test_a_warm_process_without_a_job_waits_then_exits_when_its_parent_does(tmp_path):
    import subprocess
    import sys
    import time

    parent = subprocess.Popen(["sleep", "2"])
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
    from types import SimpleNamespace

    from rewardsmith.sim import parallel
    from rewardsmith.sim.base import Stopped
    from rewardsmith.sim.progress import ProgressWriter

    procs = []

    def slow_start(source, out_dir, *args):
        parallel._prepare(source, out_dir)
        ProgressWriter(out_dir, 100).point(SimpleNamespace(steps=50, wall_s=1.0, episode_reward=2.0, terms={"a": 1.0}))
        procs.append(subprocess.Popen(["sleep", "30"]))
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
    parallel._hand_over(warm, SOURCE, tmp_path / "warm", "smoke", None, behavior_metrics=True)
    warm[0].wait()
    assert json.loads(warm[1].read_text())["behavior_metrics"] is True
