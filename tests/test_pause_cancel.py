"""Pausing, resuming and cancelling runs: the state changes and the worker's checkpoints."""

from dataclasses import replace
from pathlib import Path

import pytest

from rewardsmith import config, db
from rewardsmith.agents.fake import FakeChecker, FakeCritic, FakeEditor, FakeScener, FakeWriter
from rewardsmith.orchestrator import Cancelled, Paused, run_request
from rewardsmith.sim.fake import FakeSimulator


def _settings(tmp_path: Path) -> config.Settings:
    return config.Settings(
        db_path=tmp_path / "t.db",
        artifacts_dir=tmp_path / "artifacts",
        model="claude-opus-5-5",
        env_name="Go1JoystickFlatTerrain",
        candidates_per_round=3,
        max_rounds=3,
        success_score=99,  # never stop early, so all rounds run
        fake_sim=True,
        fake_llm=True,
    )


def _queued(tmp_path: Path, text: str = "walk backwards"):
    s = _settings(tmp_path)
    conn = db.connect(s.db_path)
    return s, conn, db.create_request(conn, text)


def test_pause_after_round_one_then_resume_finishes_the_run(tmp_path):
    s, conn, rid = _queued(tmp_path, "walk to the red ball")
    calls = {"scene": 0, "spec": 0}

    class CountingScener(FakeScener):
        def make(self, request):
            calls["scene"] += 1
            return super().make(request)

    class CountingChecker(FakeChecker):
        def write_spec(self, request):
            calls["spec"] += 1
            return super().write_spec(request)

    class PausingCritic(FakeCritic):
        def review(self, request, result, check=None):
            db.pause_request(conn, rid)  # pressed while round 1 is being judged
            return super().review(request, result, check)

    agents = dict(scener=CountingScener(), checker=CountingChecker())
    with pytest.raises(Paused):
        run_request(conn, s, db.claim_next_request(conn), FakeWriter(), FakeEditor(), PausingCritic(),
                    FakeSimulator(), **agents)
    db.finish_paused(conn, rid, 1)
    paused = db.get_request(conn, rid)
    assert paused["status"] == "paused" and paused["control"] is None
    assert [r["n"] for r in paused["rounds"]] == [1]  # stopped at the boundary, no half round
    assert db.claim_next_request(conn) is None  # paused runs stay out of the queue

    assert db.resume_request(conn, rid) == "queued"
    best = run_request(conn, s, db.claim_next_request(conn), FakeWriter(), FakeEditor(), FakeCritic(),
                       FakeSimulator(), **agents)

    done = db.get_request(conn, rid)
    assert [r["n"] for r in done["rounds"]] == [1, 2, 3]
    assert all(len(r["candidates"]) == 3 for r in done["rounds"])
    assert all(c["diff"] for c in done["rounds"][1]["candidates"])  # round 2 revised round 1's best
    assert calls == {"scene": 1, "spec": 1}  # the resumed run kept its scene and its check
    assert best is not None and done["best_score"] == best.score
    messages = [e["message"] for e in done["events"]]
    assert "Paused after round 1. Resume to continue." in messages
    assert "Resuming at round 2" in messages


def test_resume_keeps_the_best_reward_from_before_the_pause(tmp_path):
    s, conn, rid = _queued(tmp_path)
    s = replace(s, max_rounds=2)
    seen_parents = []

    class PausingCritic(FakeCritic):
        def review(self, request, result, check=None):
            db.pause_request(conn, rid)
            return super().review(request, result, check)

    class WatchingEditor(FakeEditor):
        def revise(self, request, env_doc, best_source, critique, metrics, k):
            seen_parents.append(best_source)
            return super().revise(request, env_doc, best_source, critique, metrics, k)

    with pytest.raises(Paused):
        run_request(conn, s, db.claim_next_request(conn), FakeWriter(), FakeEditor(), PausingCritic(), FakeSimulator())
    db.finish_paused(conn, rid, 1)
    round_one = db.get_request(conn, rid)["rounds"][0]
    best_source = next(c["reward_src"] for c in round_one["candidates"] if c["id"] == round_one["best_candidate_id"])

    db.resume_request(conn, rid)
    run_request(conn, s, db.claim_next_request(conn), FakeWriter(), WatchingEditor(), FakeCritic(), FakeSimulator())
    assert seen_parents[0] == best_source


def test_cancel_during_drafting_stops_before_training(tmp_path):
    s, conn, rid = _queued(tmp_path)
    trained = []

    class CancellingWriter(FakeWriter):
        def write(self, request, env_doc, k):
            db.cancel_request(conn, rid)
            return super().write(request, env_doc, k)

    class WatchingSimulator(FakeSimulator):
        def run_many(self, jobs, scene=None, cull=None):
            trained.append(len(jobs))
            return super().run_many(jobs, scene, cull)

    with pytest.raises(Cancelled):
        run_request(conn, s, db.claim_next_request(conn), CancellingWriter(), FakeEditor(), FakeCritic(),
                    WatchingSimulator())
    db.finish_cancelled(conn, rid)

    detail = db.get_request(conn, rid)
    assert trained == []  # no GPU time spent after the cancel
    assert detail["status"] == "cancelled" and detail["control"] is None
    assert detail["rounds"][0]["status"] == "done"
    assert {c["status"] for c in detail["rounds"][0]["candidates"]} == {"stopped"}
    assert detail["rounds"][0]["candidates"][0]["error"] == db.CANCELLED


def test_cancel_during_training_skips_the_critic(tmp_path):
    s, conn, rid = _queued(tmp_path)
    reviewed = []

    class CancellingSimulator(FakeSimulator):
        def run_many(self, jobs, scene=None, cull=None):
            db.cancel_request(conn, rid)
            return super().run_many(jobs, scene, cull)

    class WatchingCritic(FakeCritic):
        def review(self, request, result, check=None):
            reviewed.append(1)
            return super().review(request, result, check)

    with pytest.raises(Cancelled):
        run_request(conn, s, db.claim_next_request(conn), FakeWriter(), FakeEditor(), WatchingCritic(),
                    CancellingSimulator())
    db.finish_cancelled(conn, rid)
    assert reviewed == []
    assert {c["status"] for c in db.get_request(conn, rid)["rounds"][0]["candidates"]} == {"stopped"}


def test_state_changes(tmp_path):
    s, conn, rid = _queued(tmp_path)
    assert db.resume_request(conn, rid) is None  # nothing to resume
    assert db.pause_request(conn, rid) == "paused"  # queued: pauses at once
    assert db.pause_request(conn, rid) is None
    assert db.resume_request(conn, rid) == "queued"

    db.claim_next_request(conn)  # now running
    assert db.pause_request(conn, rid) == "pausing"
    assert db.pause_request(conn, rid) is None  # already pausing
    assert db.resume_request(conn, rid) == "running"  # withdraws the pending pause
    assert db.control_of(conn, rid) is None
    assert db.pause_request(conn, rid) == "pausing"
    assert db.cancel_request(conn, rid) == "cancelling"  # cancel overrides a pending pause
    assert db.control_of(conn, rid) == "cancel"
    assert db.cancel_request(conn, rid) is None
    db.finish_cancelled(conn, rid)
    for change in (db.pause_request, db.resume_request, db.cancel_request):
        assert change(conn, rid) is None  # nothing changes a cancelled run

    other = db.create_request(conn, "spin")
    assert db.pause_request(conn, other) == "paused"
    assert db.cancel_request(conn, other) == "cancelled"  # paused runs cancel at once


def test_a_worker_restart_leaves_paused_runs_alone(tmp_path):
    s, conn, rid = _queued(tmp_path)
    db.pause_request(conn, rid)
    assert db.fail_interrupted(conn) == []
    assert db.get_request(conn, rid)["status"] == "paused"
