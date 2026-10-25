import json
from pathlib import Path

import pytest

from rewardsmith.agents import editor, references, writer
from rewardsmith.sim import go1_spec, scene

SCENES = Path(__file__).parent.parent / "examples" / "scenes"
# Each family's reference reward and the scene it was proven on.
PROVEN = {
    "push": "ball_to_zone.json",
    "reach": "blue_box.json",
    "rough": "rough.json",
    "spin": None,
}


def _doc(name):
    return go1_spec.describe(scene.parse(json.loads((SCENES / name).read_text())) if name else scene.EMPTY)


@pytest.mark.parametrize("request_text, scene_file, fam", [
    ("push the ball into the goal zone", "ball_to_zone.json", "push"),
    ("walk to the blue box and stop beside it", "blue_box.json", "reach"),
    ("walk to the red ball", "red_ball.json", "reach"),
    ("walk forward on rough terrain", "rough.json", "rough"),
    ("spin in place", None, "spin"),
    ("turn left on the spot", None, "spin"),
    ("walk backwards", None, None),
])
def test_family_is_read_off_the_scene_and_request(request_text, scene_file, fam):
    assert references.family(request_text, _doc(scene_file)) == fam


def test_reference_section_holds_the_example_source():
    section = references.reference("push the ball into the goal zone", _doc("ball_to_zone.json"))
    assert "Reference reward for moving an object into a goal" in section
    assert (references.EXAMPLES / "push_to_zone.py").read_text().strip() in section
    assert references.reference("walk backwards", _doc(None)) == ""


def test_writer_and_editor_prompts_carry_the_reference_and_recipe(monkeypatch):
    seen = []

    def fake_ask(client, model, system, content, schema, **kw):
        seen.append((system, content[0]["text"]))
        return {"candidates": [{"idea": "x", "source": "def reward(s, a, p): return {}"}]}

    monkeypatch.setattr(writer, "ask_json", fake_ask)
    monkeypatch.setattr(editor, "ask_json", fake_ask)
    doc = _doc("ball_to_zone.json")
    writer.ClaudeWriter(None, "claude-opus-5-5").write("push the ball into the goal zone", doc, 1)
    editor.ClaudeEditor(None, "claude-opus-5-5").revise("push the ball into the goal zone", doc, "src", "c", {}, 1)
    for system, text in seen:
        assert "target_to_goal_speed" in system and "Reference reward for moving an object" in text


@pytest.mark.parametrize("fam", sorted(PROVEN))
def test_every_reference_builds_on_its_scene(fam):
    pytest.importorskip("mujoco_playground")
    from rewardsmith.sim import go1

    spec = scene.parse(json.loads((SCENES / PROVEN[fam]).read_text())) if PROVEN[fam] else scene.EMPTY
    source = (references.EXAMPLES / references.FAMILIES[fam][0]).read_text()
    env = go1.make_env(source, impl="jax", scene=spec)
    assert env.term_names
