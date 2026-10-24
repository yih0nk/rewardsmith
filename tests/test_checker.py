import math
from dataclasses import replace
from pathlib import Path

import pytest

from rewardsmith import config, db
from rewardsmith.agents import checker, critic
from rewardsmith.agents.fake import FakeChecker, FakeCritic, FakeEditor, FakeWriter
from rewardsmith.orchestrator import run_request
from rewardsmith.sim.base import RunResult
from rewardsmith.sim.fake import FakeSimulator

BACKWARDS = [{"metric": "forward_velocity", "op": "<", "value": -0.4}, {"metric": "fell", "op": "<", "value": 0.5}]


# --- validate -------------------------------------------------------------------

def test_a_good_spec_validates_to_floats():
    spec = checker.validate([{"metric": "forward_velocity", "op": "<", "value": -1}, checker.NO_FALL])
    assert spec == [{"metric": "forward_velocity", "op": "<", "value": -1.0}, {"metric": "fell", "op": "<", "value": 0.5}]
    assert isinstance(spec[0]["value"], float)


@pytest.mark.parametrize("bad, why", [
    ([], "1 to 6"),
    ([checker.NO_FALL] * 7, "1 to 6"),
    ("fell < 0.5", "1 to 6"),
    ([{"metric": "speed", "op": ">", "value": 1}], "unknown metric 'speed'"),
    ([{"metric": "fell", "op": "<=", "value": 0.5}], "op must be"),
    ([{"metric": "fell", "op": "<", "value": math.nan}], "finite number"),
    ([{"metric": "fell", "op": "<", "value": True}], "finite number"),
    ([{"metric": "fell", "op": "<", "value": "0.5"}], "finite number"),
    ([{"metric": "forward_velocity", "op": ">", "value": 30}], "sensible range"),
    ([{"metric": "fell", "op": "<", "value": 0.5, "why": "x"}], "exactly metric, op and value"),
])
def test_bad_specs_are_rejected_with_a_reason(bad, why):
    with pytest.raises(checker.SpecError, match=why):
        checker.validate(bad)


def test_schema_and_prompt_list_every_known_metric():
    enum = checker.SCHEMA["properties"]["clauses"]["items"]["properties"]["metric"]["enum"]
    assert enum == list(checker.METRICS)
    assert all(name in checker.SYSTEM for name in checker.METRICS)


# --- evaluate -------------------------------------------------------------------

def test_evaluate_reports_every_clause():
    result = checker.evaluate(BACKWARDS, {"forward_velocity": -0.3, "fell": 0.0})
    assert result["pass"] is False
    assert [(c["actual"], c["pass"]) for c in result["clauses"]] == [(-0.3, False), (0.0, True)]
    assert checker.evaluate(BACKWARDS, {"forward_velocity": -0.6, "fell": 0.0})["pass"] is True


def test_abs_metrics_are_derived_for_either_direction():
    spec = [{"metric": "abs_yaw_rate", "op": ">", "value": 1.0}]
    assert checker.evaluate(spec, {"yaw_rate": -1.4})["pass"]
    assert checker.evaluate(spec, {"yaw_rate": 1.4})["pass"]
    assert not checker.evaluate(spec, {"yaw_rate": 0.2})["pass"]


def test_missing_or_nan_metric_fails_its_clause():
    result = checker.evaluate(BACKWARDS, {"forward_velocity": math.nan})
    assert result["pass"] is False
    assert [c["actual"] for c in result["clauses"]] == [None, None]
    assert "got None, FAIL" in checker.describe(result)


# --- Claude and fake checkers ---------------------------------------------------

def test_claude_checker_retries_once_with_the_error(monkeypatch):
    prompts = []
    answers = iter([{"clauses": [{"metric": "speed", "op": ">", "value": 1}]}, {"clauses": BACKWARDS}])

    def fake_ask(client, model, system, content, schema, **kw):
        prompts.append(content[0]["text"])
        assert model == "claude-opus-5-5" and schema is checker.SCHEMA
        return next(answers)

    monkeypatch.setattr(checker, "ask_json", fake_ask)
    spec = checker.ClaudeChecker(client=None, model="claude-opus-5-5").write_spec("walk backwards")
    assert spec == BACKWARDS
    assert "walk backwards" in prompts[0]
    assert "unknown metric 'speed'" in prompts[1]


def test_claude_checker_gives_up_after_the_retry(monkeypatch):
    monkeypatch.setattr(checker, "ask_json", lambda *a, **k: {"clauses": []})
    with pytest.raises(checker.SpecError):
        checker.ClaudeChecker(client=None, model="claude-opus-5-5").write_spec("walk backwards")


@pytest.mark.parametrize("request_text, core", [
    ("walk backwards", "forward_velocity"),
    ("spin in place", "abs_yaw_rate"),
    ("run forward fast", "forward_velocity"),
    ("stand on your hind legs", "upright"),
])
def test_fake_checker_writes_valid_specs(request_text, core):
    spec = FakeChecker().write_spec(request_text)
    assert checker.validate(spec) == spec
    assert spec[0]["metric"] == core and spec[-1] == checker.NO_FALL


def test_critic_sees_the_check(monkeypatch):
    seen = {}

    def fake_ask(client, model, system, content, schema, **kw):
        seen["text"] = content[-1]["text"]
        return {"score": 7, "reward_hacking": False, "critique": "ok"}

    monkeypatch.setattr(critic, "ask_json", fake_ask)
    result = RunResult(metrics={"forward_velocity": -0.2, "fell": 0.0})
    check = checker.evaluate(BACKWARDS, result.metrics)
    critic.ClaudeCritic(client=None, model="claude-opus-5-5").review("walk backwards", result, check)
    assert "Success check (written before training):\nFAIL overall" in seen["text"]
    assert "- forward_velocity < -0.4: got -0.2, FAIL" in seen["text"]

    critic.ClaudeCritic(client=None, model="claude-opus-5-5").review("walk backwards", result)
    assert "Success check" not in seen["text"]


# --- the loop -------------------------------------------------------------------

def _settings(tmp_path: Path, **kw) -> config.Settings:
    s = config.Settings(
        db_path=tmp_path / "t.db", artifacts_dir=tmp_path / "artifacts", model="claude-opus-5-5",
        env_name="Go1JoystickFlatTerrain", candidates_per_round=2, max_rounds=3, success_score=0,
        fake_sim=True, fake_llm=True, min_rounds=1,
    )
    return replace(s, **kw)


class SpecChecker:
    def __init__(self, spec):
        self.spec, self.calls = spec, 0

    def write_spec(self, request):
        self.calls += 1
        return self.spec


class RecordingCritic(FakeCritic):
    def __init__(self):
        self.checks = []

    def review(self, request, result, check=None):
        self.checks.append(check)
        return super().review(request, result, check)


def _run(tmp_path, checker_, critic_=None, **kw):
    s = _settings(tmp_path, **kw)
    conn = db.connect(s.db_path)
    rid = db.create_request(conn, "walk backwards")
    req = db.claim_next_request(conn)
    best = run_request(conn, s, req, FakeWriter(), FakeEditor(), critic_ or FakeCritic(), FakeSimulator(), checker=checker_)
    return conn, rid, best


def test_spec_is_stored_before_training_and_every_candidate_is_checked(tmp_path):
    critic_ = RecordingCritic()
    conn, rid, best = _run(tmp_path, SpecChecker([checker.NO_FALL]), critic_)

    detail = db.get_request(conn, rid)
    assert detail["success_spec"] == [checker.NO_FALL]
    kinds = [e["kind"] for e in detail["events"]]
    assert kinds.index("check") < kinds.index("train")
    cands = [c for r in detail["rounds"] for c in r["candidates"]]
    assert cands and all(c["check_result"]["pass"] for c in cands)
    assert critic_.checks == [c["check_result"] for c in cands]
    assert detail["succeeded"] == 1


def test_a_failed_check_means_no_success_however_high_the_score(tmp_path):
    impossible = [{"metric": "forward_velocity", "op": ">", "value": 2.9}]
    conn, rid, best = _run(tmp_path, SpecChecker(impossible))

    detail = db.get_request(conn, rid)
    assert len(detail["rounds"]) == 3  # success_score 0 would have stopped after round 1
    assert detail["succeeded"] == 0
    assert best.passed is False


def test_a_candidate_that_passes_beats_a_higher_score_that_fails(tmp_path):
    # FakeSimulator's forward_velocity is a hash of the source: pick a cutoff between the two drafts.
    sources = FakeWriter().write("walk backwards", "", 2)
    fwd = [FakeSimulator().run(src, tmp_path / f"probe{i}").metrics["forward_velocity"] for i, src in enumerate(sources)]
    quality = [FakeSimulator().run(src, tmp_path / f"probe{i}").metrics["fake_quality"] for i, src in enumerate(sources)]
    low, high = sorted(fwd)
    spec = [{"metric": "forward_velocity", "op": ">", "value": (low + high) / 2}]
    passing = fwd.index(high)
    if quality[passing] > quality[1 - passing]:
        spec = [{"metric": "forward_velocity", "op": "<", "value": (low + high) / 2}]
        passing = 1 - passing
    assert quality[passing] < quality[1 - passing]  # the passing draft scores lower

    conn, rid, best = _run(tmp_path, SpecChecker(spec), max_rounds=1)
    rnd = db.get_request(conn, rid)["rounds"][0]
    assert rnd["best_candidate_id"] == rnd["candidates"][passing]["id"]
    assert best.passed and best.score == quality[passing]


def test_an_existing_spec_is_never_rewritten(tmp_path):
    s = _settings(tmp_path)
    conn = db.connect(s.db_path)
    rid = db.create_request(conn, "walk backwards")
    assert db.set_success_spec(conn, rid, BACKWARDS)
    assert not db.set_success_spec(conn, rid, [checker.NO_FALL])  # the db refuses a second spec
    req = db.claim_next_request(conn)
    other = SpecChecker([checker.NO_FALL])

    run_request(conn, s, req, FakeWriter(), FakeEditor(), FakeCritic(), FakeSimulator(), checker=other)

    assert other.calls == 0
    assert db.get_request(conn, rid)["success_spec"] == BACKWARDS


def test_a_broken_checker_falls_back_to_not_falling(tmp_path):
    class Broken:
        def write_spec(self, request):
            raise checker.SpecError("nope")

    conn, rid, _ = _run(tmp_path, Broken())
    detail = db.get_request(conn, rid)
    assert detail["success_spec"] == [checker.NO_FALL]
    assert any("Could not write the success check" in e["message"] for e in detail["events"])


def test_without_a_checker_the_loop_works_as_before(tmp_path):
    conn, rid, best = _run(tmp_path, None)
    detail = db.get_request(conn, rid)
    assert detail["success_spec"] is None
    assert all(c["check_result"] is None for r in detail["rounds"] for c in r["candidates"])
    assert detail["succeeded"] == 1 and best.passed


def test_api_exposes_the_spec_and_the_results(tmp_path, monkeypatch):
    import importlib

    from fastapi.testclient import TestClient

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("REWARDSMITH_HOME", str(tmp_path))
    import rewardsmith.server as server

    importlib.reload(server)
    client = TestClient(server.app)
    s = _settings(tmp_path, db_path=tmp_path / "rewardsmith.db")
    conn = db.connect(s.db_path)
    rid = db.create_request(conn, "walk backwards")
    run_request(conn, s, db.claim_next_request(conn), FakeWriter(), FakeEditor(), FakeCritic(), FakeSimulator(),
                checker=SpecChecker(BACKWARDS))

    detail = client.get(f"/api/requests/{rid}").json()
    assert detail["success_spec"] == BACKWARDS
    assert detail["succeeded"] in (0, 1)
    cand = detail["rounds"][0]["candidates"][0]
    assert set(cand["check_result"]) == {"pass", "clauses"}
    assert [c["metric"] for c in cand["check_result"]["clauses"]] == ["forward_velocity", "fell"]
    assert client.get("/api/requests").json()[0]["success_spec"] == BACKWARDS
