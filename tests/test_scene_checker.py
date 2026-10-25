from dataclasses import replace

import pytest

from rewardsmith import config, db
from rewardsmith.agents import checker
from rewardsmith.agents.base import Review
from rewardsmith.agents.fake import FakeChecker, FakeEditor, FakeScener, FakeWriter
from rewardsmith.orchestrator import run_request
from rewardsmith.sim.base import RunResult
from rewardsmith.sim.fake import FakeSimulator
from rewardsmith.sim.scene import parse


REACH = [
    {"metric": "target_reached", "op": ">=", "value": 0.8},
    {"metric": "target_fell", "op": "<=", "value": 0.1},
]


def target(kind="ball", movable=False):
    return parse({"objects": [{"name": kind, "kind": kind, "color": "red",
                              "size": [0.5] if kind == "zone" else [0.15],
                              "distance": 2, "movable": movable}], "target": kind})


@pytest.mark.parametrize("kind,request_text", [("ball", "walk to the red ball"), ("zone", "enter the red zone")])
def test_fake_checker_requires_reaching_and_surviving_across_episodes(kind, request_text):
    spec = FakeChecker().write_spec(request_text, scene=target(kind))
    assert spec == REACH
    assert checker.evaluate(spec, {"target_reached": 0.8, "target_fell": 0.1})["pass"]
    assert not checker.evaluate(spec, {"target_reached": 0.79, "target_fell": 0.0})["pass"]
    assert not checker.evaluate(spec, {"target_reached": 1.0, "target_fell": 0.11})["pass"]
    assert not checker.evaluate(spec, {"fell": 0.0, "upright": 1.0})["pass"]


@pytest.mark.parametrize("value", [None, float("nan"), float("inf"), True])
def test_missing_or_invalid_reach_rate_cannot_pass(value):
    assert not checker.evaluate(REACH, {"target_reached": value, "target_fell": 0.0})["pass"]


def test_claude_gets_scene_and_retries_a_check_missing_the_task(monkeypatch):
    seen = []
    answers = iter([{"clauses": [checker.NO_FALL]}, {"clauses": REACH}])
    def ask(client, model, system, content, schema, **kwargs):
        seen.append(content[0]["text"])
        return next(answers)
    monkeypatch.setattr(checker, "ask_json", ask)
    spec = checker.ClaudeChecker(None, "test").write_spec("walk to the red ball", scene=target())
    assert spec == REACH
    assert "target object is ball" in seen[0] and "red ball" in seen[0]
    assert "positive lower bound covering reach" in seen[1]


def test_explicitly_unsupported_task_is_not_retried_or_simplified(monkeypatch):
    calls = []
    def ask(*args, **kwargs):
        calls.append(1)
        return {"clauses": [], "unsupported_reason": "box-in-zone measurement is unavailable"}
    monkeypatch.setattr(checker, "ask_json", ask)
    with pytest.raises(checker.SpecError, match="Unsupported task"):
        checker.ClaudeChecker(None, "test").write_spec("push the box into the zone", scene=target())
    assert len(calls) == 1


@pytest.mark.parametrize("scene", [None, target(movable=True)])
def test_target_metrics_require_a_fixed_scene_target(scene):
    with pytest.raises(checker.SpecError, match="fixed"):
        checker.validate_for_scene(REACH, "walk to the ball", scene)


@pytest.mark.parametrize("spec", [
    [checker.NO_FALL],
    [{"metric": "target_reached", "op": "<", "value": 0.1}],
    [{"metric": "target_reached", "op": ">=", "value": 0.0}],
    [REACH[0]],
])
def test_reach_validation_rejects_trivial_or_incomplete_checks(spec):
    with pytest.raises(checker.SpecError):
        checker.validate_for_scene(spec, "walk to the ball", target())


class HighCritic:
    def review(self, request, result, check=None):
        return Review(9.0, "A high score must not bypass the task check")


def run(tmp_path, metrics, text="walk to the red ball", checker_=None):
    class MeasuredSimulator(FakeSimulator):
        def run_many(self, jobs, scene=None, cull=None):
            return [RunResult(metrics=m) for m in metrics[:len(jobs)]]
    settings = replace(config.load(), db_path=tmp_path / "test.db", artifacts_dir=tmp_path / "artifacts",
                       max_rounds=2, min_rounds=1, candidates_per_round=len(metrics), success_score=8)
    conn = db.connect(settings.db_path)
    rid = db.create_request(conn, text)
    best = run_request(conn, settings, db.claim_next_request(conn), FakeWriter(), FakeEditor(), HighCritic(),
                       MeasuredSimulator(), checker=checker_ or FakeChecker(), scener=FakeScener())
    return db.get_request(conn, rid), best


def test_scene_is_passed_once_before_training_and_passing_candidate_wins(tmp_path):
    class RecordingChecker(FakeChecker):
        scenes = []
        def write_spec(self, request, scene=None):
            self.scenes.append(scene)
            return super().write_spec(request, scene)
    agent = RecordingChecker()
    detail, best = run(tmp_path, [
        {"target_reached": 0.0, "target_fell": 0.0, "fell": 0.0},
        {"target_reached": 0.9, "target_fell": 0.0, "forward_velocity": 0.0, "fell": 0.0},
    ], checker_=agent)
    assert len(agent.scenes) == 1 and agent.scenes[0].target == "ball"
    assert detail["success_spec"] == REACH and detail["succeeded"] == 1
    assert best.candidate_id == detail["rounds"][0]["candidates"][1]["id"]
    assert len(detail["rounds"]) == 1
    kinds = [event["kind"] for event in detail["events"]]
    assert kinds.index("scene") < kinds.index("check") < kinds.index("train")


def test_stationary_robot_does_not_stop_early_despite_a_high_critic_score(tmp_path):
    detail, best = run(tmp_path, [{"target_reached": 0.0, "target_fell": 0.0, "fell": 0.0}])
    assert len(detail["rounds"]) == 2 and detail["succeeded"] == 0 and not best.passed


def test_push_task_requires_the_push_outcome_not_just_reaching(tmp_path):
    detail, best = run(tmp_path, [{"target_reached": 1.0, "target_fell": 0.0, "fell": 0.0}],
                       text="push the blue box into the green zone")
    assert detail["success_check_error"] is None
    assert detail["success_spec"][0]["metric"] == "push:box:zone"
    assert detail["succeeded"] == 0 and not best.passed


@pytest.mark.parametrize("request_text", ["walk slowly to the ball", "approach the ball", "touch the ball", "head to the ball"])
def test_common_reaching_phrases_require_outcome_metrics(request_text):
    with pytest.raises(checker.SpecError, match="target_reached"):
        checker.validate_for_scene([checker.NO_FALL], request_text, target())


def test_unsupported_task_without_a_generated_scene_still_cannot_pass():
    with pytest.raises(checker.SpecError, match="missing an object"):
        FakeChecker().write_spec("push the box into the zone")


def test_old_stored_spec_is_preserved_but_cannot_verify_a_reach_request(tmp_path):
    settings = replace(config.load(), db_path=tmp_path / "old.db", artifacts_dir=tmp_path / "artifacts",
                       max_rounds=1, min_rounds=1, candidates_per_round=1, success_score=0)
    conn = db.connect(settings.db_path)
    rid = db.create_request(conn, "walk to the ball", target().to_dict())
    db.set_success_spec(conn, rid, [checker.NO_FALL])
    run_request(conn, settings, db.claim_next_request(conn), FakeWriter(), FakeEditor(), HighCritic(),
                FakeSimulator(), checker=FakeChecker())
    detail = db.get_request(conn, rid)
    assert detail["success_spec"] == [checker.NO_FALL]
    assert detail["succeeded"] == 0
    assert "target_reached" in detail["success_check_error"]


@pytest.mark.parametrize("text,metric", [
    ("push the blue box into the green zone", "push:box:zone"),
    ("climb onto the step then stand still", "sequence:climb:box>stand_on:box"),
    ("walk to the green zone while avoiding the blue box", "all:avoid:box&reach:zone"),
    ("walk to the red ball then enter the green zone", "sequence:reach:ball>reach:zone"),
    ("enter the green zone after reaching the red ball", "sequence:reach:ball>reach:zone"),
    ("walk forward then stop", "sequence:forward>stand_still"),
    ("climb onto the step then stand still for 2 seconds", "sequence:climb:box>stand_on:box@2"),
    ("walk to the red ball then walk to the blue ball", "sequence:reach:red_ball>reach:blue_ball"),
])
def test_common_tasks_compile_to_scene_outcomes(text, metric):
    scene = parse(FakeScener().make(text))
    spec = FakeChecker().write_spec(text, scene)
    assert spec == [{"metric": metric, "op": ">=", "value": 0.8},
                    {"metric": "scene_fell", "op": "<=", "value": 0.1}]
    assert checker.evaluate(spec, {metric: 0.9, "scene_fell": 0.0})["pass"]
    assert not checker.evaluate(spec, {"target_reached": 1.0, "fell": 0.0})["pass"]


def test_independent_reach_and_avoid_rates_cannot_certify_a_combined_task():
    scene = parse(FakeScener().make("walk to the zone while avoiding the box"))
    with pytest.raises(checker.SpecError, match="together in each episode"):
        checker.validate_for_scene([
            {"metric": "reach:zone", "op": ">=", "value": 0.8},
            {"metric": "avoid:box", "op": ">=", "value": 0.8},
            {"metric": "scene_fell", "op": "<=", "value": 0.1},
        ], "walk to the zone while avoiding the box", scene)


def test_sequence_is_required_when_order_is_requested():
    scene = parse(FakeScener().make("walk to the ball then enter the zone"))
    with pytest.raises(checker.SpecError, match="in a sequence"):
        checker.validate_for_scene([
            {"metric": "all:reach:ball&reach:zone", "op": ">=", "value": 0.8},
            {"metric": "scene_fell", "op": "<=", "value": 0.1},
        ], "walk to the ball then enter the zone", scene)


def test_advanced_success_spec_is_installed_before_training(tmp_path):
    class RecordingSimulator(FakeSimulator):
        installed = None
        def set_evaluation_spec(self, spec):
            self.installed = spec
            super().set_evaluation_spec(spec)
        def run_many(self, jobs, scene=None, cull=None):
            assert self.installed[0]["metric"] == "push:box:zone"
            return [RunResult(metrics={"push:box:zone": 0.9, "scene_fell": 0.0}) for _ in jobs]
    settings = replace(config.load(), db_path=tmp_path / "push.db", artifacts_dir=tmp_path / "artifacts",
                       max_rounds=2, min_rounds=1, candidates_per_round=1, success_score=8)
    conn = db.connect(settings.db_path)
    rid = db.create_request(conn, "push the box into the zone")
    run_request(conn, settings, db.claim_next_request(conn), FakeWriter(), FakeEditor(), HighCritic(),
                RecordingSimulator(), checker=FakeChecker(), scener=FakeScener())
    detail = db.get_request(conn, rid)
    assert detail["succeeded"] == 1 and len(detail["rounds"]) == 1


def test_requested_holding_time_cannot_be_shortened():
    scene = parse(FakeScener().make("climb onto the step then stand still for 2 seconds"))
    with pytest.raises(checker.SpecError, match="holding duration"):
        checker.validate_for_scene([
            {"metric": "sequence:climb:box>stand_on:box", "op": ">=", "value": 0.8},
            {"metric": "scene_fell", "op": "<=", "value": 0.1},
        ], "climb onto the step then stand still for 2 seconds", scene)


def test_real_checker_can_return_a_concrete_push_plan(monkeypatch):
    scene = parse(FakeScener().make("push the box into the zone"))
    expected = [{"metric": "push:box:zone", "op": ">=", "value": 0.8},
                {"metric": "scene_fell", "op": "<=", "value": 0.1}]
    def ask(client, model, system, content, schema, **kwargs):
        assert "push:box:zone" in content[0]["text"]
        assert "sequence:" in system and "not two" in system
        return {"clauses": expected, "unsupported_reason": None}
    monkeypatch.setattr(checker, "ask_json", ask)
    assert checker.ClaudeChecker(None, "test").write_spec("push the box into the zone", scene) == expected


def test_checker_runs_while_the_writer_writes_and_settles_before_training(tmp_path):
    import threading
    from dataclasses import replace

    from rewardsmith import config, db
    from rewardsmith.agents.fake import FakeCritic, FakeEditor, FakeWriter
    from rewardsmith.orchestrator import run_request
    from rewardsmith.sim.fake import FakeSimulator
    from rewardsmith.sim import scene as scene_mod

    writing = threading.Event()
    spec = [{"metric": "target_reached", "op": ">=", "value": 0.8},
            {"metric": "target_fell", "op": "<=", "value": 0.1}]

    class SlowChecker:
        def write_spec(self, request, scene=None):
            # Run before the writer (scene -> checker -> writer, one after another), this times out.
            if not writing.wait(timeout=5):
                raise TimeoutError("the writer never started while the check was being written")
            self.scene = scene
            return spec

    class WaitingWriter(FakeWriter):
        def write(self, request, env_doc, k, **kw):
            writing.set()
            return super().write(request, env_doc, k, **kw)

    class Recording(FakeSimulator):
        def run_many(self, jobs, scene=None, cull=None):
            self.spec_at_training = self.evaluation_spec
            return super().run_many(jobs, scene, cull)

    s = config.Settings(
        db_path=tmp_path / "t.db", artifacts_dir=tmp_path / "artifacts", model="claude-opus-5-5",
        env_name="Go1JoystickFlatTerrain", candidates_per_round=2, max_rounds=1, success_score=0,
        fake_sim=True, fake_llm=True, min_rounds=1,
    )
    conn = db.connect(s.db_path)
    box = scene_mod.parse({"objects": [{"name": "box", "kind": "box", "color": "blue", "size": [0.4, 0.4, 0.3],
                                        "distance": [1.5, 3]}]})
    rid = db.create_request(conn, "walk to the box", scene=box.to_dict())
    checker_, sim = SlowChecker(), Recording()
    run_request(conn, s, db.claim_next_request(conn), WaitingWriter(), FakeEditor(), FakeCritic(), sim,
                checker=checker_)
    detail = db.get_request(conn, rid)
    assert detail["success_check_error"] is None and checker_.scene == box
    assert detail["success_spec"] == spec and sim.spec_at_training == spec
    kinds = [e["kind"] for e in detail["events"]]
    assert kinds.index("check") < kinds.index("train")  # still settled before anything trains
