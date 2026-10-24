from rewardsmith.agents import critic
from rewardsmith.sim.base import RunResult


def test_fall_caps_the_score():
    review = critic.apply_limits(8.5, False, "walks backwards", {"fell": 1.0})
    assert review.score == 3.0
    assert review.critique.startswith("Capped at 3: the robot fell.")


def test_reward_hacking_caps_the_score():
    review = critic.apply_limits(9.0, True, "vibrates in place", {"fell": 0.0})
    assert review.score == 3.0
    assert "Reward hacking suspected." in review.critique


def test_clean_run_keeps_its_score_and_clamps_to_range():
    assert critic.apply_limits(7.5, False, "ok", {"fell": 0.0}).score == 7.5
    assert critic.apply_limits(12.0, False, "ok", {}).score == 10.0
    assert critic.apply_limits(-1.0, False, "ok", {}).score == 0.0


def test_review_applies_limits_to_claudes_answer(monkeypatch):
    monkeypatch.setattr(critic, "ask_json", lambda *a, **k: {"score": 9, "reward_hacking": False, "critique": "fast"})
    result = RunResult(metrics={"fell": 1.0, "forward_velocity": -0.8}, frames=[], reward_curve=[1.0, 2.0])
    review = critic.ClaudeCritic(client=None, model="claude-opus-5-5").review("walk backwards", result)
    assert review.score == 3.0


def test_prompt_states_the_rubric_and_limits():
    assert "A first attempt that merely works is a 6 or 7" in critic.SYSTEM
    assert "fell" in critic.SYSTEM and "at most 3" in critic.SYSTEM
