from pathlib import Path

import pytest

from rewardsmith import config, db
from rewardsmith.agents import scener
from rewardsmith.agents.fake import FakeCritic, FakeEditor, FakeScener, FakeWriter
from rewardsmith.orchestrator import run_request
from rewardsmith.sim import scene
from rewardsmith.sim.fake import FakeSimulator

BALL = {"name": "ball", "kind": "ball", "color": "red", "size": [0.15], "distance": [1.5, 3.0],
        "bearing": [-180, 180], "yaw": [0, 0], "movable": False, "mass": 1.0}


def _settings(tmp_path: Path) -> config.Settings:
    return config.Settings(
        db_path=tmp_path / "t.db", artifacts_dir=tmp_path / "artifacts", model="claude-opus-5-5",
        env_name="Go1JoystickFlatTerrain", candidates_per_round=2, max_rounds=1, min_rounds=1,
        success_score=99, fake_sim=True, fake_llm=True,
    )


class RecordingWriter(FakeWriter):
    def write(self, request, env_doc, k):
        self.env_doc = env_doc
        return super().write(request, env_doc, k)


def _run(tmp_path, text, scener_=None, given_scene=None, writer=None):
    s = _settings(tmp_path)
    conn = db.connect(s.db_path)
    rid = db.create_request(conn, text, given_scene)
    req = db.claim_next_request(conn)
    writer = writer or RecordingWriter()
    run_request(conn, s, req, writer, FakeEditor(), FakeCritic(), FakeSimulator(), scener_)
    return db.get_request(conn, rid), writer


def test_claude_scener_returns_a_spec(monkeypatch):
    seen = {}

    def fake_ask(client, model, system, content, schema, **kw):
        seen.update(system=system, prompt=content[0]["text"], schema=schema)
        return {"plan": "one ball", "terrain": "flat", "objects": [BALL], "target": ""}

    monkeypatch.setattr(scener, "ask_json", fake_ask)
    spec = scener.ClaudeScener(client=None, model="claude-opus-5-5").make("walk to the red ball")
    assert spec == {"terrain": "flat", "objects": [BALL], "target": None}
    assert "walk to the red ball" in seen["prompt"]
    assert scene.parse(spec).target == "ball"


def test_scener_prompt_and_schema_match_the_validator():
    for word in [*scene.KINDS, *scene.COLORS, *scene.TERRAINS]:
        assert word in scener.SYSTEM
    item = scener.SCHEMA["properties"]["objects"]["items"]
    assert set(item["properties"]) == set(scene.OBJECT_KEYS)


@pytest.mark.parametrize(
    "text, names, movable, terrain",
    [
        ("walk to the red ball, then circle it", ["ball"], False, "flat"),
        ("walk backwards", [], False, "flat"),
        ("push the blue box into the green zone", ["box", "zone"], True, "flat"),
        ("stand in the yellow circle on rough ground", ["zone"], False, "rough"),
    ],
)
def test_fake_scener_reads_keywords(text, names, movable, terrain):
    spec = scene.parse(FakeScener().make(text))
    assert [o.name for o in spec.objects] == names
    assert any(o.movable for o in spec.objects) == movable
    assert spec.terrain == terrain


def test_loop_makes_and_stores_a_scene(tmp_path):
    detail, writer = _run(tmp_path, "walk to the red ball", FakeScener())
    assert detail["scene"]["objects"][0]["color"] == "red"
    assert detail["scene"]["target"] == "ball"
    assert 'state["target_dist"]' in writer.env_doc and "ball: red ball" in writer.env_doc
    cand = detail["rounds"][0]["candidates"][0]
    assert "target_dist" in cand["reward_src"]  # the fake writer used the target features
    assert "target_reached" in cand["metrics"]
    assert any(e["kind"] == "scene" and "red ball" in e["message"] for e in detail["events"])


def test_a_given_scene_skips_the_scene_writer(tmp_path):
    class Boom:
        def make(self, request):
            raise AssertionError("scene writer should not run")

    given = scene.parse({"objects": [BALL]}).to_dict()
    detail, writer = _run(tmp_path, "walk to the ball", Boom(), given_scene=given)
    assert detail["scene"] == given
    assert "ball_dist" in writer.env_doc


def test_rejected_scene_is_fixed_with_the_errors(tmp_path):
    class Sloppy:
        def __init__(self):
            self.errors = []

        def make(self, request):
            return {"objects": [{**BALL, "color": "chartreuse"}]}

        def fix(self, request, spec, error):
            self.errors.append(error)
            return {"objects": [BALL]}

    sloppy = Sloppy()
    detail, _ = _run(tmp_path, "walk to the ball", sloppy)
    assert "color must be one of" in sloppy.errors[0]
    assert detail["scene"]["objects"][0]["color"] == "red"
    assert any("rejected" in e["message"] for e in detail["events"] if e["kind"] == "scene")


def test_unfixable_scene_falls_back_to_the_plain_floor(tmp_path):
    class Hopeless:
        def make(self, request):
            return {"terrain": "lava"}

        def fix(self, request, spec, error):
            return spec

    detail, writer = _run(tmp_path, "walk on lava", Hopeless())
    assert detail["scene"] == scene.EMPTY.to_dict()
    assert "Per object" not in writer.env_doc
    assert detail["rounds"][0]["candidates"][0]["status"] == "scored"


def test_no_scener_means_no_scene(tmp_path):
    detail, writer = _run(tmp_path, "walk backwards")
    assert detail["scene"] is None
    assert "Scene:" not in writer.env_doc
