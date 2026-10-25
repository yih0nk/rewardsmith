"""Honest refusals: the judge-style prompt list, the fake classifier and the refused request."""

import json
from pathlib import Path

import pytest

from rewardsmith import config, db
from rewardsmith.agents import feasibility, scener
from rewardsmith.agents.fake import FakeCritic, FakeEditor, FakeScener, FakeWriter
from rewardsmith.orchestrator import Refused, run_request
from rewardsmith.sim.fake import FakeSimulator

PROMPTS = json.loads((Path(__file__).parent / "data" / "feasibility_prompts.json").read_text())


def test_the_prompt_list_covers_every_class():
    classes = [p["expected"] for p in PROMPTS]
    assert len(PROMPTS) >= 30
    assert {c: classes.count(c) >= 5 for c in ("doable", "approximate", "refuse")} == \
        {"doable": True, "approximate": True, "refuse": True}


@pytest.mark.parametrize("case", PROMPTS, ids=[p["prompt"] for p in PROMPTS])
def test_fake_classifier_matches_the_expected_class(case):
    assert feasibility.classify(case["prompt"])["verdict"] == case["expected"]


def test_scener_schema_carries_the_verdict():
    verdict = scener.SCHEMA["properties"]["feasibility"]
    assert verdict["properties"]["verdict"]["enum"] == ["doable", "approximate", "refuse"]
    assert "refuse" in scener.SYSTEM and "approximate" in scener.SYSTEM


def _settings(tmp_path: Path) -> config.Settings:
    return config.Settings(
        db_path=tmp_path / "t.db", artifacts_dir=tmp_path / "artifacts", model="m",
        env_name="Go1JoystickFlatTerrain", candidates_per_round=2, max_rounds=1, min_rounds=1,
        success_score=99, fake_sim=True, fake_llm=True,
    )


class CountingWriter(FakeWriter):
    calls = 0

    def write(self, request, env_doc, k, on_idea=None):
        CountingWriter.calls += 1
        return super().write(request, env_doc, k)


def test_a_refused_request_ends_before_any_reward_is_written(tmp_path):
    s = _settings(tmp_path)
    conn = db.connect(s.db_path)
    rid = db.create_request(conn, "pick up the ball with your mouth")
    req = db.claim_next_request(conn)
    CountingWriter.calls = 0
    with pytest.raises(Refused) as caught:
        run_request(conn, s, req, CountingWriter(), FakeEditor(), FakeCritic(), FakeSimulator(), scener=FakeScener())
    assert CountingWriter.calls == 0
    detail = db.get_request(conn, rid)
    assert detail["feasibility"]["verdict"] == "refuse" and detail["feasibility"]["suggestion"]
    assert detail["rounds"] == []
    assert any(e["kind"] == "refusal" for e in detail["events"])
    assert "Refused" in caught.typename


def test_an_approximate_request_trains_and_says_what_was_simplified(tmp_path):
    s = _settings(tmp_path)
    conn = db.connect(s.db_path)
    rid = db.create_request(conn, "go to the kitchen")
    req = db.claim_next_request(conn)
    run_request(conn, s, req, FakeWriter(), FakeEditor(), FakeCritic(), FakeSimulator(), scener=FakeScener())
    detail = db.get_request(conn, rid)
    assert detail["feasibility"]["verdict"] == "approximate"
    assert any(e["kind"] == "approximate" and "zone" in e["message"] for e in detail["events"])
    assert detail["rounds"]
