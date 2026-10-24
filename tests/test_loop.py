import pytest
import time
from dataclasses import replace
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
    # numbered from 1, like the board; any(): gates run concurrently
    assert any("Candidate 1, attempt 1 failed: SyntaxError" in e for e in gate_events)
    assert not any(e.startswith("Candidate 0,") for e in gate_events)


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


def test_candidates_are_marked_training_before_results_come_back(tmp_path):
    s = replace(_settings(tmp_path), max_rounds=1, min_rounds=1)
    conn = db.connect(s.db_path)
    rid = db.create_request(conn, "walk backwards")
    seen: list[set[str]] = []

    class WatchingSimulator(FakeSimulator):
        def run_many(self, jobs, scene=None):
            cands = db.get_request(conn, rid)["rounds"][-1]["candidates"]
            seen.append({c["status"] for c in cands})
            return super().run_many(jobs)

    run_request(conn, s, db.claim_next_request(conn), FakeWriter(), FakeEditor(), FakeCritic(), WatchingSimulator())

    assert seen == [{"training"}]
    detail = db.get_request(conn, rid)
    assert {c["status"] for c in detail["rounds"][0]["candidates"]} == {"scored"}
    # and the fake left a finished progress file in each candidate's folder
    from rewardsmith.sim.progress import read_progress

    assert all(read_progress(s.artifacts_dir / rid / f"r1c{i}")["stage"] == "done" for i in range(3))


class _Barrier:
    """Counts callers inside a call at once; each call waits briefly so overlaps show."""

    def __init__(self):
        import threading

        self.lock, self.inside, self.peak = threading.Lock(), 0, 0

    def __enter__(self):
        with self.lock:
            self.inside += 1
            self.peak = max(self.peak, self.inside)
        time.sleep(0.2)

    def __exit__(self, *exc):
        with self.lock:
            self.inside -= 1


def test_gates_and_critic_calls_run_concurrently(tmp_path):
    s = replace(_settings(tmp_path), max_rounds=1, min_rounds=1)
    gates, reviews = _Barrier(), _Barrier()

    class SlowGateSimulator(FakeSimulator):
        def gate(self, source, scene=None):
            with gates:
                return super().gate(source, scene)

    class SlowCritic(FakeCritic):
        def review(self, request, result):
            with reviews:
                return super().review(request, result)

    conn = db.connect(s.db_path)
    rid = db.create_request(conn, "walk backwards")
    best = run_request(conn, s, db.claim_next_request(conn), FakeWriter(), FakeEditor(), SlowCritic(), SlowGateSimulator())

    assert gates.peak == s.candidates_per_round
    assert reviews.peak == s.candidates_per_round
    # the round's best is still the top score, with ties going to the lowest index
    cands = db.get_request(conn, rid)["rounds"][0]["candidates"]
    top = max(c["score"] for c in cands)
    assert best.candidate_id == min((c for c in cands if c["score"] == top), key=lambda c: c["idx"])["id"]


def test_later_rounds_revise_the_top_two_candidates(tmp_path):
    s = replace(_settings(tmp_path), candidates_per_round=4, max_rounds=2, min_rounds=2)
    calls: list[tuple[str, int]] = []

    class RecordingEditor(FakeEditor):
        def revise(self, request, env_doc, best_source, critique, metrics, k):
            calls.append((best_source, k))
            return super().revise(request, env_doc, best_source, critique, metrics, k)

    conn = db.connect(s.db_path)
    rid = db.create_request(conn, "walk backwards")
    run_request(conn, s, db.claim_next_request(conn), FakeWriter(), RecordingEditor(), FakeCritic(), FakeSimulator())

    first = sorted(db.get_request(conn, rid)["rounds"][0]["candidates"], key=lambda c: (-c["score"], c["idx"]))
    assert sorted(calls) == sorted([(first[0]["reward_src"], 2), (first[1]["reward_src"], 2)])
    assert len(db.get_request(conn, rid)["rounds"][1]["candidates"]) == 4


def test_carry_top_one_revises_only_the_best(tmp_path):
    s = replace(_settings(tmp_path), candidates_per_round=3, max_rounds=2, min_rounds=2, carry_top=1)
    calls: list[int] = []

    class RecordingEditor(FakeEditor):
        def revise(self, request, env_doc, best_source, critique, metrics, k):
            calls.append(k)
            return super().revise(request, env_doc, best_source, critique, metrics, k)

    conn = db.connect(s.db_path)
    db.create_request(conn, "walk backwards")
    run_request(conn, s, db.claim_next_request(conn), FakeWriter(), RecordingEditor(), FakeCritic(), FakeSimulator())
    assert calls == [3]


def _start(tmp_path: Path, text: str = "walk backwards"):
    s = _settings(tmp_path)
    conn = db.connect(s.db_path)
    db.create_request(conn, text)
    return s, conn, db.claim_next_request(conn)


def test_a_critic_failure_loses_one_candidate_not_the_run(tmp_path):
    class FlakyCritic(FakeCritic):
        def review(self, request, result, check=None):
            if result.metrics["forward_velocity"] == first_velocity[0]:
                raise ConnectionError("API unreachable")
            return super().review(request, result, check)

    first_velocity = [None]

    class RememberingSimulator(FakeSimulator):
        def run_many(self, jobs, scene=None, cull=None):
            results = super().run_many(jobs, scene, cull)
            if first_velocity[0] is None:
                first_velocity[0] = results[0].metrics["forward_velocity"]
            return results

    s, conn, req = _start(tmp_path)
    best = run_request(conn, s, req, FakeWriter(), FakeEditor(), FlakyCritic(), RememberingSimulator())

    detail = db.get_request(conn, req["id"])
    assert best is not None and len(detail["rounds"]) == s.max_rounds
    first = detail["rounds"][0]["candidates"][0]
    assert first["status"] == "failed"
    assert "critic could not score" in first["error"] and "API unreachable" in first["error"]
    assert first["metrics"]  # the training result is kept
    assert any(c["status"] == "scored" for c in detail["rounds"][0]["candidates"][1:])


def test_an_editor_failure_keeps_the_rounds_that_worked(tmp_path):
    class BrokenEditor(FakeEditor):
        def revise(self, *args, **kwargs):
            raise TimeoutError("editor timed out")

    s, conn, req = _start(tmp_path)
    best = run_request(conn, s, req, FakeWriter(), BrokenEditor(), FakeCritic(), FakeSimulator())

    detail = db.get_request(conn, req["id"])
    assert best is not None and detail["best_score"] == best.score
    assert [r["status"] for r in detail["rounds"]] == ["done", "done"]
    assert detail["rounds"][1]["candidates"] == []
    assert "Stopped before round 2: TimeoutError: editor timed out" in detail["events"][-1]["message"]


def test_a_first_round_writer_failure_still_fails_the_request(tmp_path):
    class BrokenWriter(FakeWriter):
        def write(self, request, env_doc, k):
            raise ConnectionError("API unreachable")

    s, conn, req = _start(tmp_path)
    try:
        run_request(conn, s, req, BrokenWriter(), FakeEditor(), FakeCritic(), FakeSimulator())
    except ConnectionError:
        pass
    else:
        raise AssertionError("expected the writer's error to reach the worker")


def test_a_failed_fix_marks_only_that_candidate(tmp_path):
    class BadThenBrokenFix(FakeWriter):
        def write(self, request, env_doc, k):
            good = super().write(request, env_doc, k)
            return ["def nope(:"] + good[1:]

        def fix(self, source, error, env_doc):
            raise ConnectionError("API unreachable")

    s, conn, req = _start(tmp_path)
    best = run_request(conn, s, req, BadThenBrokenFix(), FakeEditor(), FakeCritic(), FakeSimulator())

    first_round = db.get_request(conn, req["id"])["rounds"][0]["candidates"]
    assert first_round[0]["status"] == "gate_failed"
    assert "The writer could not fix it: ConnectionError" in first_round[0]["error"]
    assert all(c["status"] == "scored" for c in first_round[1:])
    assert best is not None


def test_a_scene_writer_failure_falls_back_to_the_plain_floor(tmp_path):
    from rewardsmith.agents.fake import FakeScener

    class BrokenScener(FakeScener):
        def make(self, request):
            raise ConnectionError("API unreachable")

    s, conn, req = _start(tmp_path, "walk to the red ball")
    best = run_request(conn, s, req, FakeWriter(), FakeEditor(), FakeCritic(), FakeSimulator(), scener=BrokenScener())

    detail = db.get_request(conn, req["id"])
    assert best is not None
    assert any("Could not write the scene (ConnectionError" in e["message"] for e in detail["events"])


def test_failing_a_request_closes_its_unfinished_work(tmp_path):
    s, conn, req = _start(tmp_path)
    rid = req["id"]
    rnd = db.add_round(conn, rid, 1)
    statuses = ["written", "training", "trained", "scored", "gate_failed", "stopped"]
    for idx, status in enumerate(statuses):
        cid = db.add_candidate(conn, rnd, idx, "def reward(state, action, prev_action): ...")
        db.set_candidate(conn, cid, status=status)

    db.fail_request(conn, rid, "The critic is down")

    detail = db.get_request(conn, rid)
    assert detail["status"] == "failed"
    assert detail["rounds"][0]["status"] == "done"
    after = [c["status"] for c in detail["rounds"][0]["candidates"]]
    assert after == ["failed", "failed", "failed", "scored", "gate_failed", "stopped"]
    assert "before this candidate finished" in detail["rounds"][0]["candidates"][0]["error"]
    assert detail["events"][-1]["message"] == "The critic is down"


def test_fail_interrupted_also_closes_unfinished_candidates(tmp_path):
    s, conn, req = _start(tmp_path)
    rnd = db.add_round(conn, req["id"], 1)
    cid = db.add_candidate(conn, rnd, 0, "def reward(state, action, prev_action): ...")
    db.set_candidate(conn, cid, status="training")

    assert db.fail_interrupted(conn) == [req["id"]]
    assert db.get_request(conn, req["id"])["rounds"][0]["candidates"][0]["status"] == "failed"


@pytest.mark.parametrize("failure", ["writer", "editor", "gates"])
def test_unused_warm_pool_closes_on_early_exit(tmp_path, failure):
    class TrackingSimulator(FakeSimulator):
        active = False
        closed = 0
        def warm_up(self, n):
            self.active = True
        def close_warm_pool(self):
            self.active = False
            self.closed += 1
        def gate(self, source, scene=None):
            return "invalid" if failure == "gates" else super().gate(source, scene)

    class Writer(FakeWriter):
        def write(self, *args):
            if failure == "writer":
                raise ConnectionError("writer down")
            return super().write(*args)

    class Editor(FakeEditor):
        def revise(self, *args):
            if failure == "editor":
                raise ConnectionError("editor down")
            return super().revise(*args)

    s, conn, req = _start(tmp_path)
    sim = TrackingSimulator()
    if failure == "writer":
        with pytest.raises(ConnectionError):
            run_request(conn, s, req, Writer(), Editor(), FakeCritic(), sim)
    else:
        run_request(conn, s, req, Writer(), Editor(), FakeCritic(), sim)
    assert sim.closed >= 1 and sim.active is False


def test_stages_record_server_side_start_and_end_times(tmp_path):
    from rewardsmith.agents.fake import FakeChecker, FakeScener

    s = replace(_settings(tmp_path), max_rounds=2)
    conn = db.connect(s.db_path)
    rid = db.create_request(conn, "walk backwards")
    req = db.claim_next_request(conn)
    before = time.time()

    run_request(
        conn, s, req, FakeWriter(), FakeEditor(), FakeCritic(), FakeSimulator(),
        checker=FakeChecker(), scener=FakeScener(),
    )

    detail = db.get_request(conn, rid)
    names = [(st["name"], st["round_n"]) for st in detail["stages"]]
    assert names == [
        ("scene", None), ("check", None),
        ("writing", 1), ("compiling", 1), ("training", 1), ("judging", 1),
        ("writing", 2), ("compiling", 2), ("training", 2), ("judging", 2),
    ]
    for st in detail["stages"]:
        assert before <= st["started_at"] <= st["ended_at"] <= time.time()
    assert detail["stage"] is None  # nothing open once the request is done
    for cand in detail["rounds"][0]["candidates"]:
        assert cand["status"] == "scored" and cand["status_at"] >= cand["created_at"]


def test_current_stage_is_the_latest_open_one(tmp_path):
    conn = db.connect(tmp_path / "t.db")
    rid = db.create_request(conn, "walk backwards")
    with db.stage(conn, rid, "writing", 1):
        with db.stage(conn, rid, "compiling", 1) as sid:
            assert db.get_request(conn, rid)["stage"]["id"] == sid
            assert db.list_requests(conn)[0]["stage"]["name"] == "compiling"
        assert db.get_request(conn, rid)["stage"]["name"] == "writing"
    assert db.get_request(conn, rid)["stage"] is None
    assert db.list_requests(conn)[0]["stage"] is None

    db.start_stage(conn, rid, "training", 1)
    db.fail_request(conn, rid, "boom")  # a failed run must not leave a timer running
    assert db.get_request(conn, rid)["stage"] is None
