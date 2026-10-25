import json
from types import SimpleNamespace

from rewardsmith import db
from rewardsmith.agents import claude, writer
from rewardsmith.agents.stream import IdeaStream

REPLY = json.dumps({"candidates": [
    {"idea": "Reward backward speed, \"gently\".", "source": 'def reward(s, a, p):\n    return {"idea": 1.0}\n'},
    {"idea": "Track a target pose", "source": "def reward(s, a, p):\n    return {}\n"},
]})


def _chunks(text, size):
    return [text[i:i + size] for i in range(0, len(text), size)]


def test_each_idea_is_reported_once_as_soon_as_it_completes():
    for size in (1, 3, 17, len(REPLY)):
        seen = []
        stream = IdeaStream(lambda i, idea: seen.append((i, idea, len(stream.text))))
        for chunk in _chunks(REPLY, size):
            stream(chunk)
        assert [(i, idea) for i, idea, _ in seen] == [(0, 'Reward backward speed, "gently".'), (1, "Track a target pose")]
        # the first idea came well before the reply was complete, and the source's "idea" key never matched
        assert seen[0][2] < REPLY.index('"source"') + size


def test_a_failing_listener_does_not_stop_the_stream():
    calls = []

    def boom(i, idea):
        calls.append(i)
        raise RuntimeError("board down")

    stream = IdeaStream(boom)
    stream(REPLY)
    assert calls == [0, 1]


class _Stream:
    def __init__(self, text):
        self.text_stream = iter(_chunks(text, 5))
        self.final = SimpleNamespace(stop_reason="end_turn", content=[SimpleNamespace(type="text", text=text)])

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get_final_message(self):
        return self.final


def _client(text):
    return SimpleNamespace(beta=SimpleNamespace(messages=SimpleNamespace(stream=lambda **kw: _Stream(text))))


def test_the_claude_writer_streams_ideas_and_returns_the_drafts():
    seen = []
    drafts = writer.ClaudeWriter(_client(REPLY), "m").write("walk backwards", "notes", 2, on_idea=lambda i, s: seen.append(i))
    assert seen == [0, 1]
    assert [d.idea for d in drafts] == ['Reward backward speed, "gently".', "Track a target pose"]


def test_ask_json_survives_a_broken_text_listener():
    def broken(delta):
        raise ValueError("nope")

    out = claude.ask_json(_client(REPLY), "m", "sys", [], {}, on_text=broken)
    assert len(out["candidates"]) == 2


def test_the_loop_logs_ideas_for_writer_and_editor_rounds(tmp_path):
    from dataclasses import replace

    from rewardsmith.agents.fake import FakeCritic, FakeEditor, FakeWriter
    from rewardsmith.orchestrator import run_request
    from rewardsmith.sim.fake import FakeSimulator
    from tests.test_loop import _settings

    s = replace(_settings(tmp_path), max_rounds=2)
    conn = db.connect(s.db_path)
    rid = db.create_request(conn, "walk backwards")
    req = db.claim_next_request(conn)

    run_request(conn, s, req, FakeWriter(), FakeEditor(), FakeCritic(), FakeSimulator())

    events = db.get_request(conn, rid)["events"]
    ideas = [e["message"] for e in events if e["kind"] == "idea"]
    assert ideas[:3] == [f"Candidate {i + 1}: fake draft {i}: task term weighted {1.0 + i * 0.25:g}" for i in range(3)]
    # round 2 revises the top two: 2 + 1 drafts, numbered across both parents
    assert sorted(ideas[3:]) == ["Candidate 1: fake revision 0", "Candidate 2: fake revision 1", "Candidate 3: fake revision 0"]
    first_idea = next(e for e in events if e["kind"] == "idea")
    first_train = next(e for e in events if e["kind"] == "train")
    assert first_idea["id"] < first_train["id"]
