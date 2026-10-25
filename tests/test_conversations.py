import importlib
import json
import os
import sqlite3
from dataclasses import replace
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from rewardsmith import config, db, modes, orchestrator
from rewardsmith.agents.fake import FakeChecker, FakeCritic, FakeEditor, FakeScener, FakeWriter
from rewardsmith.agents.followup import FakeFollowupReader
from rewardsmith.orchestrator import prepare_turn, run_request, run_turn, training_budget
from rewardsmith.sim.fake import FakeSimulator


def _settings(tmp_path: Path, **kw) -> config.Settings:
    return replace(config.Settings(
        db_path=tmp_path / "t.db",
        artifacts_dir=tmp_path / "artifacts",
        model="claude-opus-5-5",
        env_name="Go1JoystickFlatTerrain",
        candidates_per_round=2,
        max_rounds=2,
        success_score=99,
        fake_sim=True,
        fake_llm=True,
        min_rounds=1,
    ), **kw)


class Calls:
    """Wraps an agent and records each method call's positional args."""

    def __init__(self, inner):
        self.inner, self.calls = inner, []

    def __getattr__(self, name):
        fn = getattr(self.inner, name)

        def call(*args, **kwargs):
            self.calls.append((name, args))
            return fn(*args, **kwargs)

        return call


def _first_turn(conn, s, text="push the ball to the goal"):
    rid = db.create_request(conn, text)
    req = db.claim_next_request(conn)
    run_request(conn, s, req, FakeWriter(), FakeEditor(), FakeCritic(), FakeSimulator(),
                checker=FakeChecker(), scener=FakeScener())
    db.set_request(conn, rid, status="done")
    return rid


def test_old_requests_become_one_turn_conversations(tmp_path):
    path = tmp_path / "old.db"
    old = sqlite3.connect(path)
    old.executescript(
        "CREATE TABLE requests (id TEXT PRIMARY KEY, text TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'queued',"
        " best_score REAL, created_at REAL NOT NULL);"
        "INSERT INTO requests (id, text, created_at) VALUES ('a1', 'walk backwards', 1.0), ('b2', 'spin', 2.0);"
    )
    old.commit()
    old.close()

    conn = db.connect(path)
    db.connect(path)  # a second process migrating again changes nothing
    convs = db.list_conversations(conn)
    assert [(c["id"], c["title"], [t["id"] for t in c["turns"]]) for c in convs] == [
        ("b2", "spin", ["b2"]), ("a1", "walk backwards", ["a1"])]
    assert db.get_request(conn, "a1")["turn"] == 1
    assert modes.of(db.get_request(conn, "a1")).name == modes.EFFORT  # old rows ran the full search


def test_turns_are_numbered_and_targets_must_belong(tmp_path):
    s = _settings(tmp_path)
    conn = db.connect(s.db_path)
    rid = _first_turn(conn, s)
    other = _first_turn(conn, s, "walk backwards")
    mine = db.scored_candidates(conn, rid)[0]["id"]
    theirs = db.scored_candidates(conn, other)[0]["id"]

    t2 = db.add_turn(conn, rid, "lift your feet", mine)
    t3 = db.add_turn(conn, rid, "keep going")
    assert [db.get_request(conn, t)["turn"] for t in (rid, t2, t3)] == [1, 2, 3]
    assert db.get_request(conn, t2)["conversation_id"] == rid
    with pytest.raises(db.NotInConversation):
        db.add_turn(conn, rid, "x", theirs)
    with pytest.raises(db.NotInConversation):
        db.add_turn(conn, "nope", "x")


def test_follow_up_keeps_the_scene_and_check_and_revises_the_target(tmp_path):
    s = _settings(tmp_path)
    conn = db.connect(s.db_path)
    rid = _first_turn(conn, s)
    first = db.get_request(conn, rid)
    target = db.scored_candidates(conn, rid)[-1]
    db.log_event(conn, rid, "coach", "Candidate 2 at 2.0M steps: circles the ball", round_n=target["round_n"],
                 candidate=target["idx"])

    t2 = db.add_turn(conn, rid, "get behind the ball before pushing", target["id"], mode=modes.INSTANT)
    req = db.claim_next_request(conn)
    assert req["id"] == t2
    editor, scener, checker, writer = Calls(FakeEditor()), Calls(FakeScener()), Calls(FakeChecker()), Calls(FakeWriter())
    req, follow = prepare_turn(conn, s, req, FakeFollowupReader())
    run_request(conn, modes.settings_for(s, req), req, writer, editor, FakeCritic(), FakeSimulator(),
                checker=checker, scener=scener, follow_up=follow)

    turn = db.get_request(conn, t2)
    assert turn["scene"] == first["scene"] and turn["success_spec"] == first["success_spec"]
    # The note says "before"; the check validator would read that as a new, ordered task.
    assert turn["success_check_error"] is None
    assert not scener.calls and not checker.calls  # same scene and check: no calls for them
    assert not [c for c in writer.calls if c[0] == "write"]
    (name, args), = [c for c in editor.calls if c[0] == "revise"]
    request, _, source, critique = args[:4]
    assert source == target["reward_src"]
    assert "get behind the ball before pushing" in request and "push the ball to the goal" in request
    assert "circles the ball" in critique and "get behind the ball before pushing" in critique
    assert len(turn["rounds"]) == 1 and len(turn["rounds"][0]["candidates"]) == 1  # Instant: K=1, one round
    assert turn["target_candidate_id"] == target["id"] and turn["scene_changed"] == 0
    assert turn["rounds"][0]["candidates"][0]["diff"]
    assert any("Refining Round" in e["message"] for e in turn["events"])


def test_follow_up_defaults_to_the_conversations_best(tmp_path):
    s = _settings(tmp_path)
    conn = db.connect(s.db_path)
    rid = _first_turn(conn, s)
    best = max(db.scored_candidates(conn, rid), key=lambda c: c["score"])
    db.add_turn(conn, rid, "keep going")
    req, follow = prepare_turn(conn, s, db.claim_next_request(conn), FakeFollowupReader())
    assert req["target_candidate_id"] == best["id"]
    assert follow.parent.source == best["reward_src"]
    assert req["task"] == "push the ball to the goal"  # "keep going" adds nothing to the task


def test_a_note_that_changes_the_scene_reruns_the_scene_writer(tmp_path):
    s = _settings(tmp_path)
    conn = db.connect(s.db_path)
    rid = _first_turn(conn, s, "walk forward")
    t2 = db.add_turn(conn, rid, "add a box in the way")
    req, follow = prepare_turn(conn, s, db.claim_next_request(conn), FakeFollowupReader())
    assert req["scene_changed"] == 1 and req["scene"] is None and req["success_spec"] is None
    assert follow.init_params is None
    scener = Calls(FakeScener())
    run_request(conn, s, req, FakeWriter(), FakeEditor(), FakeCritic(), FakeSimulator(), checker=FakeChecker(),
                scener=scener, follow_up=follow)
    assert scener.calls and db.get_request(conn, t2)["success_spec"]
    assert "walk forward" in scener.calls[0][1][0] and "add a box" in scener.calls[0][1][0]  # the task, not the note alone
    assert any("changes the scene" in e["message"] for e in db.get_request(conn, t2)["events"])


class RecordingSim(FakeSimulator):
    def __init__(self):
        super().__init__()
        self.jobs = []

    def run_many(self, jobs, scene=None, cull=None, on_done=None):
        self.jobs.append(list(jobs))
        return super().run_many([job[:2] for job in jobs], scene, cull, on_done)


def test_follow_up_trains_from_the_targets_weights_when_the_engine_can(tmp_path, monkeypatch):
    s = _settings(tmp_path)
    conn = db.connect(s.db_path)
    rid = _first_turn(conn, s)
    target = db.scored_candidates(conn, rid)[0]
    run_dir = s.artifacts_dir / rid / f"r{target['round_n']}c{target['idx']}"
    for steps in (3_000_000, 6_000_000):  # the latest one counts
        (run_dir / f"params_{steps}.pkl").write_bytes(b"")
    ckpt = run_dir / "params_6000000.pkl"
    db.add_turn(conn, rid, "lift your feet higher", target["id"])
    req, follow = prepare_turn(conn, s, db.claim_next_request(conn), FakeFollowupReader())
    assert follow.init_params == ckpt
    sim = RecordingSim()
    run_request(conn, replace(s, max_rounds=2, min_rounds=2, continue_steps=2_000_000), req, FakeWriter(),
                FakeEditor(), FakeCritic(), sim, follow_up=follow)
    assert all(job[2] == {"init_params": str(ckpt), "steps": 2_000_000} for job in sim.jobs[0])
    assert all(len(job) == 2 for job in sim.jobs[1])  # later rounds use the simulator's default start


def test_run_turn_applies_the_mode(tmp_path):
    s = _settings(tmp_path, candidates_per_round=3, max_rounds=3, min_rounds=2)
    conn = db.connect(s.db_path)
    rid = db.create_request(conn, "spin in place", mode=modes.INSTANT, model=modes.HAIKU)
    run_turn(conn, s, db.claim_next_request(conn), FakeSimulator())
    detail = db.get_request(conn, rid)
    assert [len(r["candidates"]) for r in detail["rounds"]] == [1]
    mode_event = next(e["message"] for e in detail["events"] if e["kind"] == "mode")
    assert mode_event.startswith("Instant · claude-haiku-5-5: 1 reward(s) a round, 1 round, 3M steps")


def test_settings_for_each_mode():
    s = config.Settings(Path("x"), Path("y"), "claude-opus-5-5", "Go1", 4, 5, 8, True, True, min_rounds=2,
                        halving=(8, 4))
    effort = modes.settings_for(s, {"mode": "effort"})
    assert (effort.candidates_per_round, effort.min_rounds, effort.max_rounds, effort.model) == (4, 1, 1, modes.OPUS)
    assert (effort.max_steps, effort.continue_steps) == (5_000_000, 2_000_000)
    assert modes.settings_for(s, {"mode": "effort", "model": modes.SONNET}).model == modes.SONNET
    instant = modes.settings_for(s, {"mode": "instant"})
    assert (instant.candidates_per_round, instant.max_rounds, instant.min_rounds, instant.model, instant.halving) == (
        1, 1, 1, modes.SONNET, ())
    assert instant.max_steps == 3_000_000
    assert modes.scene_model(s, {"mode": "instant"}) == modes.SONNET


def test_training_budget_sets_and_restores_the_step_override(monkeypatch):
    monkeypatch.setenv("REWARDSMITH_PPO_OVERRIDES", json.dumps({"num_envs": 512}))

    class Sim:  # a simulator without a per-request plan: the budget goes through the override
        preset, robustness = "round", True

    sim = Sim()
    with training_budget(sim, modes.MODES[modes.INSTANT]):
        assert json.loads(os.environ["REWARDSMITH_PPO_OVERRIDES"]) == {"num_envs": 512, "num_timesteps": 3_000_000}
        assert sim.robustness is False
    assert json.loads(os.environ["REWARDSMITH_PPO_OVERRIDES"]) == {"num_envs": 512}
    assert sim.robustness is True

    class Planned:  # the engine's simulator: run_request caps its plan, the budget restores it after
        robustness, steps = True, None

    planned = Planned()
    with training_budget(planned, modes.MODES[modes.EFFORT]):
        planned.steps = 5_000_000
        assert json.loads(os.environ["REWARDSMITH_PPO_OVERRIDES"]) == {"num_envs": 512}
    assert planned.steps is None


class PlanSim(FakeSimulator):
    """A fake with the engine's per-request plan: always 20M steps."""

    steps = None

    def plan_request(self, text, scene=None):
        self.steps = 20_000_000
        return "Training plan: complex task, 20M steps per candidate"


def test_a_mode_caps_the_simulators_plan(tmp_path):
    s = _settings(tmp_path)
    conn = db.connect(s.db_path)
    rid = db.create_request(conn, "push the ball to the goal", mode=modes.INSTANT)
    sim = PlanSim()
    run_turn(conn, s, db.claim_next_request(conn), sim)
    plan = next(e["message"] for e in db.get_request(conn, rid)["events"] if e["kind"] == "plan")
    assert plan.endswith("this mode caps it at 3M steps")
    assert sim.steps is None  # put back after the run


def _client(tmp_path, monkeypatch) -> TestClient:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("REWARDSMITH_HOME", str(tmp_path))
    import rewardsmith.server as server

    importlib.reload(server)
    return TestClient(server.app)


def test_api_conversations_and_follow_ups(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    created = client.post("/api/requests", json={"text": "walk backwards", "mode": "instant", "model": modes.HAIKU})
    assert created.status_code == 201
    rid = created.json()["id"]
    assert created.json()["conversation_id"] == rid

    turn = client.post(f"/api/conversations/{rid}/turns", json={"text": "faster"})
    assert turn.status_code == 201
    t2 = turn.json()["id"]
    empty = client.post(f"/api/conversations/{rid}/turns", json={"text": "", "mode": "effort"}).json()["id"]

    listed = {r["id"]: r for r in client.get("/api/requests").json()}
    assert (listed[rid]["mode"], listed[rid]["model"], listed[rid]["turn"]) == ("instant", modes.HAIKU, 1)
    assert (listed[t2]["mode"], listed[t2]["turn"], listed[t2]["conversation_id"]) == ("instant", 2, rid)
    assert (listed[empty]["text"], listed[empty]["mode"]) == ("Keep going", "effort")
    conv, = client.get("/api/conversations").json()
    assert [t["id"] for t in conv["turns"]] == [rid, t2, empty]

    assert client.post("/api/conversations/nope/turns", json={"text": "x"}).status_code == 404
    assert client.post(f"/api/conversations/{rid}/turns", json={"text": "x", "target_candidate_id": 999}).status_code == 404
    assert client.post("/api/requests", json={"text": "walk", "model": "gpt-5"}).status_code == 422
    assert client.post("/api/requests", json={"text": "walk", "mode": "slow"}).status_code == 422
    assert client.get("/api/requests/" + rid).json()["mode"] == "instant"
    health = client.get("/api/health").json()
    assert health["models"] == list(modes.MODELS) and health["default_followup_mode"] == "instant"


def test_the_idle_worker_warms_the_largest_round():
    calls = []

    class Sim:
        def warm_up(self, n):
            calls.append(n)

    orchestrator.prewarm(Sim(), config.Settings(Path("x"), Path("y"), "m", "Go1", 3, 2, 8, True, True))
    orchestrator.prewarm(FakeSimulator(), config.Settings(Path("x"), Path("y"), "m", "Go1", 3, 2, 8, True, True))
    assert calls == [4]  # Effort's K, the largest of the modes


def test_a_draft_with_idea_and_source_swapped_is_put_right():
    from rewardsmith.agents.base import Draft

    code = "def reward(state, action, prev_action):\n    return {'x': 1.0}\n"
    fixed = orchestrator._as_draft(Draft("Push from behind the ball, then stop in the goal.", idea=code))
    assert (fixed.source, fixed.idea) == (code, "Push from behind the ball, then stop in the goal.")
    fine = Draft(code, idea="def reward is mentioned in prose here")
    assert orchestrator._as_draft(fine) == fine
