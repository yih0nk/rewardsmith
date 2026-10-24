from pathlib import Path

from rewardsmith import config, db
from rewardsmith.agents.fake import FakeCritic, FakeEditor, FakeWriter
from rewardsmith.bench.behaviors import BEHAVIORS
from rewardsmith.orchestrator import run_request
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


def test_claim_is_exclusive(tmp_path):
    conn = db.connect(tmp_path / "t.db")
    rid = db.create_request(conn, "walk backwards")
    assert db.claim_next_request(conn)["id"] == rid
    assert db.claim_next_request(conn) is None


def test_fake_loop_runs_all_rounds(tmp_path):
    s = _settings(tmp_path)
    conn = db.connect(s.db_path)
    rid = db.create_request(conn, "walk backwards")
    req = db.claim_next_request(conn)

    best = run_request(conn, s, req, FakeWriter(), FakeEditor(), FakeCritic(), FakeSimulator())

    detail = db.get_request(conn, rid)
    assert best is not None
    assert len(detail["rounds"]) == 3
    assert all(len(r["candidates"]) == 3 for r in detail["rounds"])
    assert detail["rounds"][1]["candidates"][0]["diff"]  # revisions carry a diff
    assert detail["best_score"] == best.score


def test_runs_min_rounds_even_when_round_one_succeeds(tmp_path):
    from dataclasses import replace

    s = replace(_settings(tmp_path), success_score=0, min_rounds=2)  # round 1 already "succeeds"
    conn = db.connect(s.db_path)
    db.create_request(conn, "walk backwards")
    req = db.claim_next_request(conn)

    run_request(conn, s, req, FakeWriter(), FakeEditor(), FakeCritic(), FakeSimulator())

    rounds = db.get_request(conn, req["id"])["rounds"]
    assert len(rounds) == 2
    assert rounds[1]["candidates"][0]["diff"]  # the editor ran


def test_gate_failure_is_recorded(tmp_path):
    class BrokenWriter(FakeWriter):
        def write(self, request, env_doc, k):
            return ["def nope(:"] * k

    s = _settings(tmp_path)
    conn = db.connect(s.db_path)
    db.create_request(conn, "spin")
    req = db.claim_next_request(conn)

    best = run_request(conn, s, req, BrokenWriter(), FakeEditor(), FakeCritic(), FakeSimulator())

    detail = db.get_request(conn, req["id"])
    assert best is None
    assert {c["status"] for r in detail["rounds"] for c in r["candidates"]} == {"gate_failed"}
    gate_events = [e["message"] for e in detail["events"] if e["kind"] == "gate"]
    # every attempt is logged, not just the last: 3 rounds x 3 candidates x 3 attempts
    assert len(gate_events) == 3 * 3 * 3
    assert "Candidate 0, attempt 1 failed: SyntaxError" in gate_events[0]


def test_behavior_checks():
    walk_back = next(b for b in BEHAVIORS if b.slug == "walk_backwards")
    assert walk_back.passes({"forward_velocity": -0.5})
    assert not walk_back.passes({"forward_velocity": 0.2})
    assert not walk_back.passes({})


def test_fake_metrics_include_reward_terms(tmp_path):
    s = _settings(tmp_path)
    conn = db.connect(s.db_path)
    rid = db.create_request(conn, "walk backwards")
    run_request(conn, s, db.claim_next_request(conn), FakeWriter(), FakeEditor(), FakeCritic(), FakeSimulator())

    scored = [c for r in db.get_request(conn, rid)["rounds"] for c in r["candidates"] if c["status"] == "scored"]
    assert scored
    # FakeWriter's rewards return "task" and "energy" terms.
    assert all({"term_task", "term_energy"} <= set(c["metrics"]) for c in scored)
