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
        out = []
        for source, _ in jobs:
            m = self.metrics_by_source[source]
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
    assert [r["status"] for r in rows] == ["FAIL", "PASS", "FAIL"]
    assert rows[0]["reasons"] == ["error: RewardError: boom"]
    assert rows[2]["reasons"] == ["not run"]
    assert "FAIL  walk_backwards" in regression.format_rows(rows)


def test_record_then_compare_from_the_command_line(tmp_path, capsys):
    args = ["--baseline", str(tmp_path / "baseline.json"), "--out", str(tmp_path / "out")]
    assert regression.main(["--record", *args], sim=_good_sim()) == 0
    baseline = json.loads((tmp_path / "baseline.json").read_text())
    assert baseline["preset"] == "round"
    assert set(baseline["cases"]) == {"walk_backwards", "walk_forward", "spin_in_place"}
    assert baseline["cases"]["walk_backwards"]["compile_s"] == 40.0

    assert regression.main(["--compare", *args], sim=_good_sim()) == 0
    assert "REGRESSION PASS" in capsys.readouterr().out

    sim = _good_sim()
    sim.metrics_by_source[_source(regression.CASES[2])] = _metrics(fwd=0.0, yaw=0.2)  # stopped spinning
    assert regression.main(["--compare", *args], sim=sim) == 1
    assert "REGRESSION FAIL" in capsys.readouterr().out


def test_record_refuses_a_baseline_whose_behavior_fails(tmp_path):
    sim = _good_sim()
    sim.metrics_by_source[_source(regression.CASES[2])] = _metrics(fwd=0.0, yaw=0.2)
    args = ["--record", "--baseline", str(tmp_path / "baseline.json"), "--out", str(tmp_path / "out")]
    assert regression.main(args, sim=sim) == 1
    assert not (tmp_path / "baseline.json").exists()


def test_compare_refuses_a_different_preset(tmp_path):
    args = ["--baseline", str(tmp_path / "baseline.json"), "--out", str(tmp_path / "out")]
    regression.main(["--record", *args], sim=_good_sim())
    with pytest.raises(SystemExit):
        regression.main(["--compare", "--preset", "smoke", *args], sim=_good_sim())
