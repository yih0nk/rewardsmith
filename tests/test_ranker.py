from rewardsmith.agents import ranker
from rewardsmith.search import _as_order


def test_rank_sends_every_run_and_returns_claudes_order(monkeypatch):
    sent = {}

    def fake_ask(client, model, system, content, schema, effort):
        sent["text"], sent["effort"] = content[0]["text"], effort
        return {"order": [2, 0, 1], "reason": "run 2 already walks backwards"}

    monkeypatch.setattr(ranker, "ask_json", fake_ask)
    snaps = [{"forward_velocity": -0.1234567, "steps": 100}, {"forward_velocity": 0.2}, {"forward_velocity": -0.5}]
    order = ranker.ClaudeRanker(client=None, model="claude-opus-5-5").rank("walk backwards", snaps)

    assert order == [2, 0, 1]
    assert "run 0: " in sent["text"] and "run 2: " in sent["text"] and "-0.123" in sent["text"]
    assert sent["effort"] == "low"  # a quick call: the survivors keep training while it runs


def test_bad_orders_are_repaired_into_a_full_ranking():
    assert _as_order([2, 2, 7, -1, 0], 4) == [2, 0, 1, 3]
