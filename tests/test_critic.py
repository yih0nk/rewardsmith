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


PUSH_METRICS = {
    "fell": 0.0, "target_reached": 1.0, "object_goal_final_dist": 1.4, "push:ball:goal": 0.0,
    "phase_no_approach": 0.0, "phase_no_push": 0.75, "phase_wrong_way": 0.0, "phase_short": 0.25,
    "phase_overshoot": 0.0, "phase_delivered": 0.0,
}
PUSH_CHECK = {"pass": False, "clauses": [
    {"metric": "fell", "op": "<", "value": 0.5, "actual": 0.0, "pass": True},
    {"metric": "push:ball:goal", "op": ">=", "value": 0.8, "actual": 0.0, "pass": False},
    {"metric": "scene_fell", "op": "<=", "value": 0.1, "actual": 0.0, "pass": True},
]}


def test_failed_task_clause_caps_the_score_and_names_the_phase():
    review = critic.apply_limits(7.0, False, "walks to the ball and parks", PUSH_METRICS, PUSH_CHECK)
    assert review.score == 4.0
    assert review.critique.startswith("Capped at 4: the task was not done (push:ball:goal >= 0.8, got 0).")
    assert "fail at 'no_push' (75%)" in review.critique and "target_to_goal_speed" in review.critique
    assert review.critique.endswith("walks to the ball and parks")


def test_a_failed_safety_clause_alone_does_not_trigger_the_task_cap():
    for metric in ("upright", "scene_fell", "target_fell"):
        check = {"pass": False, "clauses": [{"metric": metric, "op": "<", "value": 0.1, "actual": 0.5, "pass": False}]}
        assert critic.apply_limits(6.0, False, "ok", {"fell": 0.0}, check).score == 6.0, metric
    assert critic.is_task_metric("target_reached") and critic.is_task_metric("sequence:reach:ball>climb:step")
    assert critic.apply_limits(3.0, False, "ok", PUSH_METRICS, PUSH_CHECK).score == 3.0  # already below


def test_diagnosis_reports_success_and_skips_plain_tasks():
    done = {"phase_no_approach": 0.0, "phase_left": 0.1, "phase_arrived": 0.9}
    assert "fail at 'left' (10%)" in critic.task_diagnosis(done) and "Done in 90%" in critic.task_diagnosis(done)
    assert "got done in 100%" in critic.task_diagnosis({"phase_no_approach": 0.0, "phase_left": 0.0,
                                                         "phase_arrived": 1.0})
    assert critic.task_diagnosis({"forward_velocity": 0.5}) is None


def test_critic_prompt_carries_the_task_diagnosis(monkeypatch):
    seen = {}

    def fake_ask(client, model, system, content, schema, **kw):
        seen["text"] = content[-1]["text"]
        return {"score": 8, "reward_hacking": False, "critique": "pushes"}

    monkeypatch.setattr(critic, "ask_json", fake_ask)
    review = critic.ClaudeCritic(client=None, model="claude-opus-5-5").review(
        "push the ball into the goal", RunResult(metrics=PUSH_METRICS), PUSH_CHECK)
    assert "Task diagnosis (computed from the 32 episodes):\nEpisodes by phase" in seen["text"]
    assert review.score == 4.0


def test_a_failed_pose_clause_caps_the_score():
    check = {"pass": False, "clauses": [
        {"metric": "front_feet_off", "op": ">=", "value": 0.6, "actual": 0.2, "pass": False},
        {"metric": "fell", "op": "<", "value": 0.5, "actual": 0.0, "pass": True},
    ]}
    review = critic.apply_limits(6.5, False, "rears up, drops back", {"fell": 0.0}, check)
    assert review.score == 4.0 and "front_feet_off >= 0.6, got 0.2" in review.critique
    assert "front_feet_off" in critic.SYSTEM
