"""Body routing: the request's body field, the router's pick, and body-aware prompts.

Fake capability cards stand in for the G1 and the Panda, so these run before those
bodies are registered.
"""

import importlib
import sqlite3
from dataclasses import replace
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from rewardsmith import config, db, orchestrator
from rewardsmith.agents import checker, router, writer
from rewardsmith.agents.robot import for_body
from rewardsmith.orchestrator import Refused, route_body, run_turn
from rewardsmith.sim import bodies
from rewardsmith.sim.bodies.base import Capabilities
from rewardsmith.sim.fake import FakeSimulator


class FakeBody:
    def __init__(self, name, title, caps, metrics=None):
        self.name, self.title, self.capabilities, self._metrics = name, title, caps, metrics or {}

    def metric_names(self):
        return self._metrics


G1 = FakeBody("g1", "Unitree G1", Capabilities(
    kind="humanoid", legs=2, can_walk=True, bipedal=True, can_grasp=False, reach_m=None,
    summary="Unitree G1: humanoid, walks on two legs, waves its arms, no gripper"),
    {"hand_height": ("mean height of the right hand above the floor, m", 0.0, 2.0)})
PANDA = FakeBody("panda", "Franka Panda", Capabilities(
    kind="arm", legs=0, can_walk=False, bipedal=False, can_grasp=True, reach_m=0.85,
    summary="Franka Panda: 7-joint arm on a table with a parallel gripper, picks and places objects"))


@pytest.fixture
def registry(monkeypatch):
    """The Go1 plus these fake bodies, for one test: registry(G1, PANDA). registry() is the Go1
    alone, whatever else is registered for real."""
    def add(*extra):
        monkeypatch.setattr(bodies, "BODIES", {"go1": bodies.get_body("go1"), **{b.name: b for b in extra}})
        return bodies.BODIES
    return add


ALL = "all"
# (prompt, bodies registered besides the Go1, expected pick: a name, or None to refuse)
CASES = [
    ("pick up the cube", (PANDA, G1), "panda"),
    ("grab the red block and place it in the bowl", (PANDA, G1), "panda"),
    ("stack the two boxes", (PANDA,), "panda"),
    ("walk on two legs", (PANDA, G1), "g1"),
    ("wave hello", (PANDA, G1), "g1"),
    ("walk like a human across the room", (G1,), "g1"),
    ("trot forward", (PANDA, G1), "go1"),
    ("push the ball with your body into the goal", (PANDA, G1), "go1"),
    ("walk backwards", (PANDA, G1), "go1"),
    ("spin in place", (G1,), "go1"),
    ("climb the stairs", (PANDA, G1), "go1"),
    ("stand on your hind legs", (PANDA, G1), "go1"),
    ("pick up the cube", (G1,), None),  # no gripper anywhere: refused
    ("wave hello", (PANDA,), None),  # no humanoid
    ("walk forward", (PANDA,), "go1"),
    ("pick up the ball", (), None),
]


@pytest.mark.parametrize("prompt,extra,expected", CASES)
def test_fake_router_picks_the_body_the_task_needs(registry, prompt, extra, expected):
    pick = router.FakeRouter().pick(prompt, router.cards(registry(*extra)))
    assert pick["body"] == expected
    assert pick["reason"]
    if expected is None:
        assert pick["suggestion"]


def test_cards_carry_every_capability(registry):
    cards = router.cards(registry(G1))
    assert [c["name"] for c in cards] == ["go1", "g1"]
    assert cards[1] == {"name": "g1", "title": "Unitree G1", "kind": "humanoid", "legs": 2, "can_walk": True,
                        "bipedal": True, "can_grasp": False, "reach_m": None, "summary": G1.capabilities.summary}


def test_claude_router_only_offers_registered_bodies_and_drops_unknown_picks(registry):
    cards = router.cards(registry(G1))
    assert router.schema([c["name"] for c in cards])["properties"]["body"]["enum"] == ["go1", "g1", "none"]
    prompt = router._prompt("wave", cards)
    assert "- g1: Unitree G1: humanoid" in prompt and "panda" not in prompt.lower()
    assert router._clean({"body": "panda", "reason": "x.", "suggestion": "wave"}, ["go1", "g1"]) == {
        "body": None, "reason": "x", "suggestion": "wave"}


def _settings(tmp_path: Path) -> config.Settings:
    return config.Settings(
        db_path=tmp_path / "t.db", artifacts_dir=tmp_path / "artifacts", model="claude-opus-5-5",
        env_name="Go1JoystickFlatTerrain", candidates_per_round=1, max_rounds=1, success_score=99,
        fake_sim=True, fake_llm=True, min_rounds=1,
    )


class Spy:
    def __init__(self, pick=None):
        self.calls, self._pick = [], pick

    def pick(self, request, cards):
        self.calls.append((request, [c["name"] for c in cards]))
        if isinstance(self._pick, Exception):
            raise self._pick
        return self._pick


def test_go1_alone_routes_without_a_call_or_an_event(tmp_path, registry):
    registry()
    conn = db.connect(tmp_path / "t.db")
    rid = db.create_request(conn, "pick up the cube")
    spy = Spy()
    assert route_body(conn, _settings(tmp_path), db.claim_next_request(conn), spy) == "go1"
    assert spy.calls == []
    req = db.get_request(conn, rid)
    assert (req["body"], req["body_reason"]) == ("go1", None)
    assert not [e for e in req["events"] if e["kind"] in ("body", "refusal")]


def test_auto_routes_and_says_why(tmp_path, registry):
    registry(G1, PANDA)
    conn = db.connect(tmp_path / "t.db")
    rid = db.create_request(conn, "wave hello")
    spy = Spy({"body": "g1", "reason": "the task needs two legs", "suggestion": ""})
    assert route_body(conn, _settings(tmp_path), db.claim_next_request(conn), spy) == "g1"
    assert spy.calls == [("wave hello", ["go1", "g1", "panda"])]
    req = db.get_request(conn, rid)
    assert (req["body"], req["body_reason"]) == ("g1", "the task needs two legs")
    assert [e["message"] for e in req["events"] if e["kind"] == "body"] == [
        "Using the Unitree G1: the task needs two legs"]
    assert [s["name"] for s in req["stages"]] == ["body"]


def test_the_users_pick_wins_and_follow_ups_inherit_it(tmp_path, registry):
    registry(G1, PANDA)
    conn = db.connect(tmp_path / "t.db")
    rid = db.create_request(conn, "walk forward", body="g1")
    spy = Spy()
    assert route_body(conn, _settings(tmp_path), db.claim_next_request(conn), spy) == "g1"
    db.set_request(conn, rid, status="done")
    t2 = db.add_turn(conn, rid, "faster")
    assert db.get_request(conn, t2)["body"] == "g1"
    assert route_body(conn, _settings(tmp_path), db.claim_next_request(conn), spy) == "g1"
    assert spy.calls == []


def test_a_follow_up_queued_while_its_task_is_on_auto_gets_the_pick(tmp_path, registry):
    registry(G1)
    conn = db.connect(tmp_path / "t.db")
    rid = db.create_request(conn, "wave")
    t2 = db.add_turn(conn, rid, "keep going")
    assert db.get_request(conn, t2)["body"] is None
    route_body(conn, _settings(tmp_path), db.claim_next_request(conn), Spy({"body": "g1", "reason": "two legs"}))
    db.set_request(conn, rid, status="done")
    assert route_body(conn, _settings(tmp_path), db.claim_next_request(conn), Spy()) == "g1"
    assert db.get_request(conn, t2)["body"] == "g1"


def test_no_body_fits_refuses_with_the_reason(tmp_path, registry):
    registry(G1)
    conn = db.connect(tmp_path / "t.db")
    rid = db.create_request(conn, "pick up the cube")
    spy = Spy({"body": None, "reason": "Picking up needs a gripper", "suggestion": "push the cube"})
    with pytest.raises(Refused) as exc:
        route_body(conn, _settings(tmp_path), db.claim_next_request(conn), spy)
    assert exc.value.feasibility["verdict"] == "refuse"
    refusal, = [e["message"] for e in db.get_request(conn, rid)["events"] if e["kind"] == "refusal"]
    assert refusal == "Can't train this: Picking up needs a gripper. Try instead: push the cube"


def test_a_failed_router_call_falls_back_to_the_go1(tmp_path, registry):
    registry(G1)
    conn = db.connect(tmp_path / "t.db")
    rid = db.create_request(conn, "wave")
    assert route_body(conn, _settings(tmp_path), db.claim_next_request(conn), Spy(RuntimeError("boom"))) == "go1"
    assert "couldn't choose a body" in db.get_request(conn, rid)["body_reason"]


def test_run_turn_sets_the_simulators_body_before_it_describes_the_env(tmp_path, registry):
    registry(G1)
    s = _settings(tmp_path)
    conn = db.connect(s.db_path)
    sim, seen = FakeSimulator(), []
    sim.describe = lambda scene=None: (seen.append(sim.body), "notes")[1]
    rid = db.create_request(conn, "walk backwards", body="go1")
    run_turn(conn, s, db.claim_next_request(conn), sim)
    assert seen == ["go1"]
    assert db.get_request(conn, rid)["best_score"] is not None
    rid2 = db.create_request(conn, "wave", body="g1")
    sim.describe = lambda scene=None: (seen.append(sim.body), "notes")[1]
    run_turn(conn, s, db.claim_next_request(conn), sim)
    assert seen == ["go1", "g1"] and sim.body == "go1"  # back to the worker's body after the run
    assert db.get_request(conn, rid2)["body"] == "g1"
    sim.set_body("g1")
    assert sim.body == "g1"
    sim.set_body(None)
    assert sim.body == "go1"


def test_old_rows_read_as_the_go1(tmp_path):
    path = tmp_path / "old.db"
    old = sqlite3.connect(path)
    old.executescript(
        "CREATE TABLE requests (id TEXT PRIMARY KEY, text TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'done',"
        " best_score REAL, created_at REAL NOT NULL);"
        "INSERT INTO requests (id, text, created_at) VALUES ('a1', 'walk backwards', 1.0);"
    )
    old.commit()
    old.close()
    conn = db.connect(path)
    db.connect(path)  # a second process migrating again changes nothing
    req = db.get_request(conn, "a1")
    assert (req["body"], req["body_reason"], req["training_steps"]) == ("go1", None, None)  # both migrations applied
    assert db.list_conversations(conn)[0]["body"] == "go1"
    assert db.get_request(conn, db.create_request(conn, "wave"))["body"] is None  # a new task on Auto


def _client(tmp_path, monkeypatch) -> TestClient:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("REWARDSMITH_HOME", str(tmp_path))
    import rewardsmith.server as server

    importlib.reload(server)
    return TestClient(server.app)


def test_api_takes_a_registered_body_or_auto(tmp_path, monkeypatch, registry):
    registry(G1)
    client = _client(tmp_path, monkeypatch)
    health = client.get("/api/health").json()
    assert health["default_body"] == "go1"
    assert [(b["name"], b["title"], b["kind"]) for b in health["bodies"]] == [
        ("go1", "Unitree Go1", "quadruped"), ("g1", "Unitree G1", "humanoid")]
    assert health["bodies"][1]["summary"].startswith("Unitree G1: humanoid")

    picked = client.post("/api/requests", json={"text": "wave hello", "body": "g1", "training_steps": 20_000_000}).json()["id"]
    auto = client.post("/api/requests", json={"text": "trot", "body": "auto"}).json()["id"]
    omitted = client.post("/api/requests", json={"text": "trot"}).json()["id"]
    assert client.post("/api/requests", json={"text": "pick it up", "body": "panda"}).status_code == 422
    listed = {r["id"]: r["body"] for r in client.get("/api/requests").json()}
    assert (listed[picked], listed[auto], listed[omitted]) == ("g1", None, None)
    assert client.get(f"/api/requests/{picked}").json()["training_steps"] == 20_000_000
    turn = client.post(f"/api/conversations/{picked}/turns", json={"text": "higher", "body": "go1"}).json()["id"]
    assert client.get(f"/api/requests/{turn}").json()["body"] == "g1"  # follow-ups keep the task's body
    conv = next(c for c in client.get("/api/conversations").json() if c["id"] == picked)
    assert conv["body"] == "g1"


def test_prompts_name_the_body_and_stay_word_for_word_on_the_go1(registry):
    registry(G1, PANDA)
    assert for_body(writer.SYSTEM, "go1") == writer.SYSTEM == writer.system_for(None)
    assert checker.system_for("go1") == checker.SYSTEM
    humanoid = checker.system_for("g1")
    assert "Unitree G1 humanoid" in humanoid and "Unitree Go1" not in humanoid
    assert "- hand_height: mean height of the right hand above the floor, m" in humanoid
    assert "front_feet_off" not in humanoid and "Pose requests" not in humanoid
    assert "about 0.28 standing" not in humanoid and "- base_height: mean height of the body above the floor, m" in humanoid
    assert checker.validate([{"metric": "base_height", "op": ">=", "value": 0.7}])
    assert checker.metric_definition("hand_height") == ("mean height of the right hand above the floor, m", 0.0, 2.0)
    arm = writer.system_for("panda")
    assert "stock gait" not in arm and "hind legs" not in arm
    assert "Keep the stock terms the environment notes list" in arm
    assert arm.rstrip().endswith("The robot: " + PANDA.capabilities.summary + ".")


def test_no_agent_prompt_calls_the_robot_a_dog():
    import pathlib
    import re

    root = pathlib.Path(orchestrator.__file__).parent / "agents"
    for path in root.glob("*.py"):
        if path.name in ("fake.py", "feasibility.py"):  # offline keyword rules, not prompts
            continue
        assert not re.search(r"robot dog", path.read_text(), re.I), path.name
