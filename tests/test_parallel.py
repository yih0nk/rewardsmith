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
