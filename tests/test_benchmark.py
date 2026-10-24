import importlib.util
from dataclasses import replace
from pathlib import Path

import pytest

from rewardsmith import config, db
from rewardsmith.agents import checker
from rewardsmith.agents.fake import FakeCritic, FakeEditor, FakeWriter
from rewardsmith.bench import report
from rewardsmith.bench.suite import BY_SLUG, CASES
from rewardsmith.orchestrator import run_request
from rewardsmith.sim.fake import FakeSimulator

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("benchmark_script", ROOT / "scripts" / "benchmark.py")
benchmark = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(benchmark)


class LearnsInRound2(FakeSimulator):
    """Walks backwards only once FakeEditor has revised the reward, i.e. from round 2."""

    def run(self, source, out_dir, scene=None):
        result = super().run(source, out_dir, scene)
        revised = "fake revision" in source
        result.metrics.update(forward_velocity=-0.5 if revised else 0.4, lateral_velocity=0.0, yaw_rate=0.0)
        result.metrics.update(compile_s=30.0, train_s=90.0)
        return result


class ExplodingChecker:
    def write_spec(self, request):
        raise AssertionError("benchmark requests carry their own check; the checker must not run")


def _settings(tmp_path: Path) -> config.Settings:
    return config.Settings(
        db_path=tmp_path / "t.db",
        artifacts_dir=tmp_path / "artifacts",
        model="claude-opus-5-5",
        env_name="Go1JoystickFlatTerrain",
        candidates_per_round=2,
        max_rounds=3,
        success_score=99,  # the critic never ends a run early: every request runs 3 rounds
        fake_sim=True,
        fake_llm=True,
    )


def _run_all(conn, settings, sim):
    while (req := db.claim_next_request(conn)) is not None:
        run_request(conn, settings, req, FakeWriter(), FakeEditor(), FakeCritic(), sim, checker=ExplodingChecker())
        db.set_request(conn, req["id"], status="done")


def test_every_measurable_spec_is_valid():
    for case in CASES:
        if case.spec is not None:
            assert checker.validate([dict(c) for c in case.spec])
    assert [c.slug for c in CASES if c.spec is None] == ["limp", "pronk"]


def test_queued_requests_keep_their_fixed_check(tmp_path):
    s = _settings(tmp_path)
    conn = db.connect(s.db_path)
    rid = benchmark.queue_case(conn, BY_SLUG["walk_backwards"])

    _run_all(conn, s, LearnsInRound2())

    detail = db.get_request(conn, rid)
    assert detail["success_spec"] == list(BY_SLUG["walk_backwards"].spec)
    checks = [c["check_result"] for r in detail["rounds"] for c in r["candidates"]]
    assert all(isinstance(c, dict) for c in checks)


def test_report_numbers_from_a_fake_benchmark(tmp_path):
    s = _settings(tmp_path)
    conn = db.connect(s.db_path)
    cases = [BY_SLUG["walk_backwards"], BY_SLUG["walk_left"], BY_SLUG["walk_forward"]]
    rids = [benchmark.queue_case(conn, c) for c in cases]

    _run_all(conn, s, LearnsInRound2())

    outcomes = [report.outcome(c, db.get_request(conn, rid), s.success_score) for c, rid in zip(cases, rids)]
    back, left, forward = outcomes
    assert back.first_pass_round == 2  # round 1 walked forwards; FakeEditor's revisions walk backwards
    assert left.first_pass_round is None
    assert forward.first_pass_round == 1
    assert back.gpu_s == pytest.approx(3 * 2 * 120)  # 3 rounds x 2 candidates x (30 + 90) s

    baseline = {"cases": {"walk_forward": {"metrics": {"forward_velocity": 0.6}}}}
    summary = report.summarize(outcomes, s.max_rounds, baseline, gpu_dollars_per_hour=0.6)

    # walk_forward is the reference, not one of the behaviors: 1 of 2 behaviors succeeds, in round 2.
    assert summary["success_rate_by_round"] == {1: 0.0, 2: 0.5, 3: 0.5}
    assert summary["median_rounds_to_success"] == 2
    assert summary["median_gpu_minutes_per_success"] == pytest.approx(12)
    assert summary["median_gpu_dollars_per_success"] == pytest.approx(0.12)
    assert summary["unmeasured"] == ["limp", "pronk"]
    assert summary["critic_judged"] == 18  # every candidate was scored and checked
    ref = summary["reference"]
    assert ref["hand_written_forward_velocity"] == 0.6
    assert ref["generated_forward_velocity"] == pytest.approx(0.4)  # the passing round-1 candidate
    assert ref["percent_of_hand_written"] == pytest.approx(100 * 0.4 / 0.6)

    text = report.markdown(summary, outcomes)
    assert "0% after round 1, 50% by round 3" in text
    assert "Not run, no metric can check them yet: limp, pronk." in text
    assert "| walk_left | done | 3 | - |" in text


def test_unverified_checks_never_count_as_success():
    case = BY_SLUG["walk_backwards"]
    detail = {
        "id": "x",
        "status": "done",
        "created_at": 0.0,
        "events": [],
        "rounds": [{"n": 1, "candidates": [
            {"id": 1, "score": 9.0, "metrics": {}, "check_result": {"pass": True, "verified": False}},
        ]}],
    }
    assert report.outcome(case, detail, 8.0).first_pass_round is None


def test_pending_requests_are_left_out_of_the_rates(tmp_path):
    case = BY_SLUG["spin"]
    running = report.Outcome(case, "a", "running", 1, None, None)
    done = report.Outcome(case, "b", "done", 3, 1, 9.0)
    summary = report.summarize([running, done], 3)
    assert summary["finished"] == 1 and summary["pending"] == 1
    assert summary["success_rate_by_round"][1] == 1.0
