import importlib.util
import json
from pathlib import Path

import pytest

from rewardsmith.sim.base import RunResult
from rewardsmith.sim.fake import FakeSimulator

ROOT = Path(__file__).parent.parent
_spec = importlib.util.spec_from_file_location("regression", ROOT / "scripts" / "regression.py")
regression = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(regression)


class StubSim:
    """Returns fixed metrics per reward file, like a trained run would."""

    def __init__(self, metrics_by_source: dict[str, dict]):
        self.metrics_by_source = metrics_by_source

    def run_many(self, jobs):
        """A list of metrics for a source gives one entry per repeat, in job order."""
        out = []
        for i, (source, _) in enumerate(jobs):
            m = self.metrics_by_source[source]
            if isinstance(m, list):
                m = m[i]
            out.append(m if isinstance(m, str) else RunResult(metrics=m))
        return out


def _metrics(reward=30.0, fwd=-0.6, yaw=0.0, fell=0.0):
    return {"episode_reward": reward, "forward_velocity": fwd, "yaw_rate": yaw, "fell": fell,
            "compile_s": 40.0, "train_s": 130.0, "upright": 0.99}


def _source(case):
    return (ROOT / case.reward).read_text()


def _run(sim, tmp_path):
    return regression.run_suite(sim, regression.CASES, tmp_path, "round", "warp")


def _good_sim():
    walk_back, walk_fwd, spin = regression.CASES
    return StubSim({
        _source(walk_back): _metrics(fwd=-0.6),
        _source(walk_fwd): _metrics(fwd=0.6),
        _source(spin): _metrics(fwd=0.05, yaw=1.4),
    })


def test_every_case_reward_file_exists_and_passes_the_gate():
    for case in regression.CASES:
        assert FakeSimulator().gate(_source(case)) is None


def test_a_good_run_passes_its_own_baseline(tmp_path):
    run = _run(_good_sim(), tmp_path)
    assert all(r["behavior_pass"] for r in run["cases"].values())
    rows = regression.compare(run, json.loads(json.dumps(run)))
    assert [r["status"] for r in rows] == ["PASS"] * 3


def test_a_failed_behavior_check_fails(tmp_path):
    baseline = _run(_good_sim(), tmp_path)
    sim = _good_sim()
    sim.metrics_by_source[_source(regression.CASES[0])] = _metrics(fwd=-0.1)  # barely moving backwards
    rows = regression.compare(_run(sim, tmp_path), baseline)
    assert [r["status"] for r in rows] == ["FAIL", "PASS", "PASS"]
    assert "forward_velocity" in rows[0]["reasons"][0]


def test_reward_more_than_5_percent_below_baseline_fails(tmp_path):
    baseline = _run(_good_sim(), tmp_path)
    sim = _good_sim()
    sim.metrics_by_source[_source(regression.CASES[1])] = _metrics(reward=28.4, fwd=0.6)  # -5.3%
    sim.metrics_by_source[_source(regression.CASES[2])] = _metrics(reward=28.6, fwd=0.0, yaw=1.4)  # -4.7%
    rows = regression.compare(_run(sim, tmp_path), baseline)
    assert [r["status"] for r in rows] == ["PASS", "FAIL", "PASS"]


def test_a_negative_baseline_reward_still_allows_5_percent(tmp_path):
    baseline = {"cases": {"walk_backwards": {"episode_reward": -10.0}}}
    ok = {"cases": {"walk_backwards": {"episode_reward": -10.4, "checks": []}}}
    bad = {"cases": {"walk_backwards": {"episode_reward": -10.6, "checks": []}}}
    assert regression.compare(ok, baseline)[0]["status"] == "PASS"
    assert regression.compare(bad, baseline)[0]["status"] == "FAIL"


def test_a_crashed_or_missing_case_fails(tmp_path):
    baseline = _run(_good_sim(), tmp_path)
    sim = _good_sim()
    sim.metrics_by_source[_source(regression.CASES[0])] = "RewardError: boom"
    run = _run(sim, tmp_path)
    del run["cases"]["spin_in_place"]
    rows = regression.compare(run, baseline)
    assert [r["status"] for r in rows] == ["FAIL", "PASS", "WARN"]  # spin is advisory
    assert rows[0]["reasons"] == ["error: RewardError: boom"]
    assert rows[2]["reasons"] == ["not run"]
    assert "FAIL  walk_backwards" in regression.format_rows(rows)


def test_record_then_compare_from_the_command_line(tmp_path, capsys):
    args = ["--baseline", str(tmp_path / "baseline.json"), "--out", str(tmp_path / "out")]
    assert regression.main(["--record", *args], sim=_good_sim()) == 0
    baseline = json.loads((tmp_path / "baseline.json").read_text())
    assert baseline["preset"] == "round" and baseline["repeats"] == 3
    assert set(baseline["cases"]) == {"walk_backwards", "walk_forward", "spin_in_place"}
    assert baseline["cases"]["walk_backwards"]["compile_s"] == 40.0

    assert regression.main(["--compare", *args], sim=_good_sim()) == 0
    assert "REGRESSION PASS" in capsys.readouterr().out

    sim = _good_sim()
    sim.metrics_by_source[_source(regression.CASES[1])] = _metrics(fwd=0.1)  # barely walking forward
    assert regression.main(["--compare", *args], sim=sim) == 1
    assert "REGRESSION FAIL" in capsys.readouterr().out


def test_a_spin_regression_warns_without_failing(tmp_path, capsys):
    args = ["--baseline", str(tmp_path / "baseline.json"), "--out", str(tmp_path / "out")]
    regression.main(["--record", *args], sim=_good_sim())
    sim = _good_sim()
    sim.metrics_by_source[_source(regression.CASES[2])] = _metrics(fwd=0.0, yaw=1.4, fell=1.0)  # fell over
    assert regression.main(["--compare", *args], sim=sim) == 0
    out = capsys.readouterr().out
    assert "WARN  spin_in_place" in out and "REGRESSION PASS (1 advisory warning)" in out


def test_record_refuses_a_baseline_whose_behavior_fails(tmp_path):
    sim = _good_sim()
    sim.metrics_by_source[_source(regression.CASES[0])] = _metrics(fwd=0.2)  # walking forward, not backwards
    args = ["--record", "--baseline", str(tmp_path / "baseline.json"), "--out", str(tmp_path / "out")]
    assert regression.main(args, sim=sim) == 1
    assert not (tmp_path / "baseline.json").exists()


def test_compare_refuses_a_different_preset(tmp_path):
    args = ["--baseline", str(tmp_path / "baseline.json"), "--out", str(tmp_path / "out")]
    regression.main(["--record", *args], sim=_good_sim())
    with pytest.raises(SystemExit):
        regression.main(["--compare", "--preset", "smoke", *args], sim=_good_sim())


def test_one_collapsed_run_of_three_is_outvoted(tmp_path):
    baseline = _run(_good_sim(), tmp_path)
    sim = _good_sim()
    spin = _source(regression.CASES[2])
    sim.metrics_by_source[spin] = [_metrics(reward=1.9, fwd=0.1, yaw=0.3, fell=1.0),  # training collapsed
                                   _metrics(reward=29.5, fwd=0.0, yaw=1.4), _metrics(reward=30.4, fwd=0.0, yaw=1.5)]
    run = _run(sim, tmp_path)
    spin_run = run["cases"]["spin_in_place"]
    assert spin_run["episode_reward"] == 29.5  # the median, not the mean
    assert spin_run["behavior_passed"] == "2/3" and spin_run["behavior_pass"]
    assert [r["status"] for r in regression.compare(run, baseline)] == ["PASS"] * 3
    assert "[runs 1.9, 29.5, 30.4]" in regression.format_rows(regression.compare(run, baseline))


def test_two_bad_runs_of_three_fail(tmp_path):
    baseline = _run(_good_sim(), tmp_path)
    sim = _good_sim()
    walk = _source(regression.CASES[0])
    sim.metrics_by_source[walk] = [_metrics(fwd=-0.6), "RuntimeError: out of memory", _metrics(fwd=-0.1)]
    rows = regression.compare(_run(sim, tmp_path), baseline)
    assert rows[0]["status"] == "FAIL"
    assert "passed 1/3" in rows[0]["reasons"][0]


def test_clauses_that_fail_in_different_runs_fail_the_behavior(tmp_path):
    baseline = _run(_good_sim(), tmp_path)
    sim = _good_sim()
    spin = _source(regression.CASES[2])
    sim.metrics_by_source[spin] = [_metrics(fwd=0.0, yaw=0.5), _metrics(fwd=0.0, yaw=1.4, fell=1.0),
                                   _metrics(fwd=0.0, yaw=1.4)]
    rows = regression.compare(_run(sim, tmp_path), baseline)
    assert rows[2]["status"] == "WARN"
    assert rows[2]["reasons"] == ["behavior check passed in only 1/3 runs"]
