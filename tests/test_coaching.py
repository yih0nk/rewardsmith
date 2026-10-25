import threading
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from rewardsmith import config, db
from rewardsmith.agents import coach as coach_mod
from rewardsmith.agents.base import Observation
from rewardsmith.agents.coach import ClaudeCoach, check_now, cost, prompt
from rewardsmith.agents.fake import FakeCoach, FakeCritic, FakeEditor, FakeWriter
from rewardsmith.coaching import ACTED, LAST_RUNNER, NOTHING, QUOTA, TOO_EARLY, LiveCoach
from rewardsmith.orchestrator import build_coach, run_request
from rewardsmith.sim.base import RunResult
from rewardsmith.sim.fake import FakeSimulator
from rewardsmith.sim.progress import ProgressWriter


def _evals(out_dir, n, total=1000, terms=None):
    w = ProgressWriter(out_dir, total)
    for i in range(n):
        w.point(SimpleNamespace(steps=i * 100, wall_s=float(i), episode_reward=float(i),
                                terms=terms or {"task": 0.1 * i}, behavior={"forward_velocity": -0.1 * i}))
    return w


class ScriptedCoach:
    """Says `verdict` for the jobs in `bad`, continue for the rest."""

    def __init__(self, bad=(), verdict="kill"):
        self.bad, self.verdict, self.seen = set(bad), verdict, []
        self.lock = threading.Lock()

    def observe(self, request, source, curve, total_steps, spec=None, frames=()):
        with self.lock:
            self.seen.append((source, len(curve)))
        verdict = self.verdict if source in self.bad else "continue"
        return Observation(verdict, f"{source} looks {verdict}", f"fix {source}", latency_s=0.5, cost_usd=0.001)


def _poll(cull, running, until, timeout=5.0):
    deadline, stops = time.monotonic() + timeout, {}
    while time.monotonic() < deadline:
        stops.update(cull(running))
        if until(stops):
            break
        time.sleep(0.01)
    return stops


def test_kills_at_eval_two_and_reports_every_verdict(tmp_path):
    dirs = [tmp_path / f"c{i}" for i in range(3)]
    for d in dirs:
        _evals(d, 3)  # evals 0, 1, 2
    notes, stopped = [], []
    live = LiveCoach(ScriptedCoach(bad={"b"}), "walk backwards", list(zip("abc", dirs)),
                     on_note=lambda call, action: notes.append((call.idx, action)),
                     on_stop=lambda i, call, metrics: stopped.append((i, metrics)))
    stop = _poll(live, {0, 1, 2}, until=lambda s: len(notes) == 3)
    assert set(stop) == {1} and "kill" in stop[1] and "b looks kill" in stop[1]
    assert sorted(notes) == [(0, NOTHING), (1, ACTED), (2, NOTHING)]
    assert stopped[0][0] == 1 and stopped[0][1]["forward_velocity"] == pytest.approx(-0.2)
    assert live.finished and live.stats()["stopped"] == 1


def test_never_kills_before_eval_two(tmp_path):
    dirs = [tmp_path / f"c{i}" for i in range(2)]
    for d in dirs:
        _evals(d, 2)  # only evals 0 and 1
    notes = []
    live = LiveCoach(ScriptedCoach(bad={"a"}), "r", list(zip("ab", dirs)), on_note=lambda c, a: notes.append(a))
    assert _poll(live, {0, 1}, until=lambda s: len(notes) == 2, timeout=1.0) == {}
    assert TOO_EARLY in notes


def test_never_kills_the_last_runner(tmp_path):
    dirs = [tmp_path / f"c{i}" for i in range(2)]
    for d in dirs:
        _evals(d, 3)
    notes = []
    live = LiveCoach(ScriptedCoach(bad={"a", "b"}), "r", list(zip("ab", dirs)), on_note=lambda c, a: notes.append(a))
    stop = _poll(live, {0, 1}, until=lambda s: len(notes) == 2, timeout=2.0)
    assert len(stop) == 1  # one goes, the other is the last one training
    assert sorted(notes) == sorted([ACTED, LAST_RUNNER])


def test_stops_at_most_half_the_round(tmp_path):
    dirs = [tmp_path / f"c{i}" for i in range(4)]
    for d in dirs:
        _evals(d, 3)
    notes = []
    live = LiveCoach(ScriptedCoach(bad=set("abc")), "r", list(zip("abcd", dirs)), on_note=lambda c, a: notes.append(a))
    stop = _poll(live, {0, 1, 2, 3}, until=lambda s: len(notes) == 4, timeout=2.0)
    assert len(stop) == 2 and sorted(notes) == sorted([ACTED, ACTED, QUOTA, NOTHING])


def test_asks_once_per_new_eval_and_skips_the_final_one(tmp_path):
    d = tmp_path / "c0"
    w = _evals(d, 2, total=300)
    coach = ScriptedCoach()
    live = LiveCoach(coach, "r", [("a", d), ("b", tmp_path / "missing")])
    _poll(live, {0, 1}, until=lambda s: live.calls and live.calls[-1].observation is not None, timeout=1.0)
    for _ in range(5):
        live({0, 1})
    assert coach.seen == [("a", 2)]
    w.point(SimpleNamespace(steps=300, wall_s=3.0, episode_reward=3.0, terms={}, behavior={}))  # the final eval
    for _ in range(5):
        live({0, 1})
    assert len(coach.seen) == 1


def test_a_run_whose_evals_landed_at_once_gets_one_look(tmp_path):
    d = tmp_path / "c0"
    w = _evals(d, 5, total=400)
    w.stage("done")
    coach = ScriptedCoach()
    live = LiveCoach(coach, "r", [("a", d), ("b", tmp_path / "x")])
    _poll(live, {0, 1}, until=lambda s: live.finished, timeout=1.0)
    assert coach.seen == [("a", 4)]  # evals 0-3: the last one is the critic's


def test_a_failing_coach_costs_a_note_not_the_run(tmp_path):
    dirs = [tmp_path / f"c{i}" for i in range(2)]
    for d in dirs:
        _evals(d, 3)

    class Broken:
        def observe(self, *a, **k):
            raise RuntimeError("overloaded")

    notes = []
    live = LiveCoach(Broken(), "r", list(zip("ab", dirs)), on_note=lambda c, a: notes.append(a))
    assert _poll(live, {0, 1}, until=lambda s: live.finished, timeout=1.0) == {}
    assert notes == [] and live.stats()["failed"] == 2


def test_halving_decides_first(tmp_path):
    dirs = [tmp_path / f"c{i}" for i in range(3)]
    for d in dirs:
        _evals(d, 3)
    coach = ScriptedCoach()
    live = LiveCoach(coach, "r", list(zip("abc", dirs)), halving=lambda running: {0: "halved"} if 0 in running else {})
    stop = live({0, 1, 2})
    assert stop == {0: "halved"}
    _poll(live, {1, 2}, until=lambda s: live.finished)
    assert sorted(s for s, _ in coach.seen) == ["b", "c"]


def test_prompt_shows_terms_behavior_and_the_check_so_far():
    curve = [{"steps": 0, "episode_reward": 0.0, "terms": {"task": 0.0}, "behavior": {"forward_velocity": 0.0}},
             {"steps": 2_000_000, "wall_s": 9.0, "episode_reward": 50.0, "terms": {"task": 0.1, "posture": 50.0},
              "behavior": {"forward_velocity": 0.05, "upright": 0.99}}]
    spec = [{"metric": "forward_velocity", "op": "<", "value": -0.3}, {"metric": "fell", "op": "<", "value": 0.5}]
    text = prompt("walk backwards", "def reward(...): ...", curve, 10_000_000, spec)
    assert "posture" in text and "2,000,000 of 10,000,000" in text
    assert "forward_velocity < -0.3: now 0.05, fails" in text
    assert "fell < 0.5: now" not in text  # not measured mid-training
    assert "wall_s" not in text
    assert [c["metric"] for c in check_now(spec, curve[-1])] == ["forward_velocity"]


def test_claude_coach_drops_the_fallback_on_haiku(monkeypatch):
    calls = []

    def fake_ask(client, model, system, content, schema, **kw):
        calls.append((model, kw))
        kw["on_usage"](SimpleNamespace(input_tokens=2000, output_tokens=200))
        return {"verdict": "revise", "note": " gaming posture ", "diagnosis": "cut posture 100x"}

    monkeypatch.setattr(coach_mod, "ask_json", fake_ask)
    obs = ClaudeCoach(None).observe("r", "src", [{"steps": 0}, {"steps": 1}], 10)
    assert obs.verdict == "revise" and obs.note == "gaming posture"
    assert calls[0][0] == "claude-haiku-5-5" and calls[0][1]["fallback"] is False and calls[0][1]["effort"] == "low"
    assert obs.cost_usd == pytest.approx(cost("claude-haiku-5-5", 2000, 200))
    ClaudeCoach(None, "claude-sonnet-5-5").observe("r", "src", [{"steps": 0}, {"steps": 1}], 10)
    assert calls[1][1]["fallback"] is True


def _settings(tmp_path: Path, **kw) -> config.Settings:
    return config.Settings(
        db_path=tmp_path / "t.db", artifacts_dir=tmp_path / "artifacts", model="m", env_name="Go1JoystickFlatTerrain",
        candidates_per_round=4, max_rounds=2, success_score=99, fake_sim=True, fake_llm=True, **kw,
    )


def _start(tmp_path, **kw):
    s = _settings(tmp_path, **kw)
    conn = db.connect(s.db_path)
    db.create_request(conn, "walk backwards")
    return s, conn, db.claim_next_request(conn)


def test_build_coach_follows_the_request_then_the_setting(tmp_path):
    s = _settings(tmp_path)
    assert build_coach(s) is None  # the dataclass default is off
    assert isinstance(build_coach(replace(s, coach="claude-haiku-5-5")), FakeCoach)
    assert build_coach(replace(s, coach="claude-haiku-5-5"), {"coach": "off"}) is None
    assert isinstance(build_coach(s, {"coach": "claude-sonnet-5-5"}), FakeCoach)


def test_fake_coach_comments_on_every_candidate(tmp_path):
    s, conn, req = _start(tmp_path)
    run_request(conn, s, req, FakeWriter(), FakeEditor(), FakeCritic(), FakeSimulator(), coach=FakeCoach())
    events = [e for e in db.get_request(conn, req["id"])["events"] if e["kind"] == "coach"]
    notes = [e for e in events if e["candidate"] is not None]
    assert {(e["round_n"], e["candidate"]) for e in notes} == {(r, c) for r in (1, 2) for c in range(4)}
    assert all(e["message"].startswith(f"Candidate {e['candidate'] + 1} at ") for e in notes)
    assert any(e["message"].startswith("Coach: 4 looks") for e in events)
    rounds = db.get_request(conn, req["id"])["rounds"]
    assert all(c["status"] == "scored" for r in rounds for c in r["candidates"])


def test_a_stopped_candidate_is_revised_into_the_next_round(tmp_path):
    s, conn, req = _start(tmp_path)
    writer = FakeWriter()
    first = writer.write("walk backwards", "", 4)
    revised = []

    class Editor(FakeEditor):
        def revise(self, request, env_doc, best_source, critique, metrics, k, on_idea=None):
            revised.append((best_source, critique, k))
            return super().revise(request, env_doc, best_source, critique, metrics, k, on_idea)

    run_request(conn, s, req, writer, Editor(), FakeCritic(), FakeSimulator(),
                coach=ScriptedCoach(bad={first[1]}, verdict="revise"))
    r1, r2 = db.get_request(conn, req["id"])["rounds"]
    assert r1["candidates"][1]["status"] == "stopped" and "revise" in r1["candidates"][1]["error"]
    assert [c["status"] for i, c in enumerate(r1["candidates"]) if i != 1] == ["scored"] * 3
    early = revised[0]
    assert early[0] == first[1] and "Live training coach" in early[1] and "fix " in early[1] and early[2] == 1
    assert sum(k for _, _, k in revised) == 4 and len(r2["candidates"]) == 4  # 1 early + 3 from the top two
    events = [e["message"] for e in db.get_request(conn, req["id"])["events"]]
    assert any("Drafting a fix for candidate 2 for round 2" in m for m in events)
    assert any("Stopping it." in m for m in events)


def test_no_early_drafts_in_the_last_round(tmp_path):
    s, conn, req = _start(tmp_path)
    s = replace(s, max_rounds=1, min_rounds=1)
    writer = FakeWriter()
    first = writer.write("walk backwards", "", 4)
    calls = []

    class Editor(FakeEditor):
        def revise(self, *a, **k):
            calls.append(1)
            return super().revise(*a, **k)

    run_request(conn, s, req, writer, Editor(), FakeCritic(), FakeSimulator(), coach=ScriptedCoach(bad={first[0]}))
    assert calls == []


def test_each_candidate_is_scored_as_soon_as_its_run_ends(tmp_path):
    s, conn, req = _start(tmp_path)
    s = replace(s, max_rounds=1, min_rounds=1)
    scored_before_last_run = []

    class Sim(FakeSimulator):
        def run_many(self, jobs, scene=None, cull=None, on_done=None):
            results = []
            for i, (src, out_dir) in enumerate(jobs):
                if i == len(jobs) - 1:  # before the last run ends, the first ones are already scored
                    deadline = time.monotonic() + 5
                    while time.monotonic() < deadline:
                        rnd = db.get_request(conn, req["id"])["rounds"][0]
                        done = [c for c in rnd["candidates"] if c["status"] == "scored"]
                        if len(done) == len(jobs) - 1:
                            break
                        time.sleep(0.02)
                    scored_before_last_run.append(len(done))
                results.append(self.run(src, out_dir, scene))
                on_done(i, results[-1])
            return results

    run_request(conn, s, req, FakeWriter(), FakeEditor(), FakeCritic(), Sim())
    assert scored_before_last_run == [3]
    stages = [st["name"] for st in db.get_request(conn, req["id"])["stages"]]
    assert stages.count("judging") == 1


def test_coach_events_and_options_reach_the_api(tmp_path, monkeypatch):
    monkeypatch.setenv("REWARDSMITH_HOME", str(tmp_path))
    import importlib

    from fastapi.testclient import TestClient

    from rewardsmith import server

    importlib.reload(server)
    client = TestClient(server.app)
    rid = client.post("/api/requests", json={"text": "walk backwards", "coach": "claude-sonnet-5-5"}).json()["id"]
    assert client.post("/api/requests", json={"text": "walk backwards", "coach": "gpt"}).status_code == 422
    conn = db.connect(server.settings.db_path)
    db.log_event(conn, rid, "coach", "Candidate 2 at 2.6M steps: standing still", round_n=1, candidate=1)
    body = client.get(f"/api/requests/{rid}").json()
    assert body["coach"] == "claude-sonnet-5-5"
    assert body["events"][-1] | {"created_at": 0, "id": 0} == {
        "id": 0, "request_id": rid, "kind": "coach", "message": "Candidate 2 at 2.6M steps: standing still",
        "created_at": 0, "round_n": 1, "candidate": 1,
    }
