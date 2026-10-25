import importlib
import json
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from rewardsmith import config, db, modes
from rewardsmith.orchestrator import run_turn, training_budget
from rewardsmith.sim.fake import FakeSimulator


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("REWARDSMITH_HOME", str(tmp_path))
    from rewardsmith import server
    return TestClient(importlib.reload(server).app)


def test_budget_is_saved_for_tasks_and_followups(client):
    rid = client.post("/api/requests", json={"text": "avoid the box", "training_steps": 50_000_000}).json()["id"]
    assert client.get(f"/api/requests/{rid}").json()["training_steps"] == 50_000_000
    follow = client.post(f"/api/conversations/{rid}/turns", json={"training_steps": 20_000_000}).json()["id"]
    assert client.get(f"/api/requests/{follow}").json()["training_steps"] == 20_000_000
    default = client.post(f"/api/conversations/{rid}/turns", json={}).json()["id"]
    assert client.get(f"/api/requests/{default}").json()["training_steps"] is None


@pytest.mark.parametrize("steps", [0, -1, 999_999, 100_000_001, True, "20000000", 20_000_000.5])
def test_invalid_budgets_are_rejected_on_both_routes(client, steps):
    rid = client.post("/api/requests", json={"text": "walk backwards"}).json()["id"]
    assert client.post("/api/requests", json={"text": "avoid the box", "training_steps": steps}).status_code == 422
    assert client.post(f"/api/conversations/{rid}/turns", json={"training_steps": steps}).status_code == 422


@pytest.mark.parametrize("steps", [1_000_000, 100_000_000])
def test_budget_bounds_are_accepted(client, steps):
    assert client.post("/api/requests", json={"text": "avoid the box", "training_steps": steps}).status_code == 201


def test_custom_budget_replaces_both_mode_caps_and_restores_override(monkeypatch):
    settings = config.Settings(Path("db"), Path("artifacts"), modes.OPUS, "Go1", 4, 5, 8, True, True)
    for name in modes.MODES:
        req = {"mode": name, "training_steps": 50_000_000}
        selected = modes.settings_for(settings, req)
        assert selected.max_steps == selected.continue_steps == 50_000_000
    assert modes.MODES[modes.INSTANT].steps == 3_000_000
    assert modes.MODES[modes.EFFORT].continue_steps == 4_000_000
    monkeypatch.setenv("REWARDSMITH_PPO_OVERRIDES", '{"num_envs":512}')
    class Sim:
        preset = "round"
    with training_budget(Sim(), modes.of(req)):
        assert json.loads(os.environ["REWARDSMITH_PPO_OVERRIDES"])["num_timesteps"] == 50_000_000
    assert json.loads(os.environ["REWARDSMITH_PPO_OVERRIDES"]) == {"num_envs": 512}


@pytest.mark.parametrize("automatic_steps", [10_000_000, None])
def test_custom_steps_raise_and_lower_the_scene_plan_without_leaking(tmp_path, automatic_steps):
    class Planned(FakeSimulator):
        steps = None
        seen = []

        def plan_request(self, text, scene=None):
            self.steps = automatic_steps
            return "Automatic scene plan: 10M steps"

        def run_many(self, *args, **kwargs):
            self.seen.append(self.steps)
            return super().run_many(*args, **kwargs)

    settings = config.Settings(tmp_path / "db", tmp_path / "artifacts", modes.SONNET,
                               "Go1JoystickFlatTerrain", 4, 1, 8, True, True)
    conn = db.connect(settings.db_path)
    sim = Planned()
    for steps, expected in [(50_000_000, 50_000_000), (2_000_000, 2_000_000), (None, 3_000_000)]:
        rid = db.create_request(conn, "walk backwards", mode="instant", training_steps=steps)
        run_turn(conn, settings, db.claim_next_request(conn), sim)
        db.set_request(conn, rid, status="done")
        assert sim.seen[-1] == expected
        assert sim.steps is None
        events = db.get_request(conn, rid)["events"]
        if steps:
            assert any(f"{steps / 1e6:g}M steps" in e["message"] for e in events if e["kind"] == "plan")


def test_existing_database_gets_nullable_budget_column(tmp_path):
    import sqlite3
    path = tmp_path / "old.db"
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE requests (id TEXT PRIMARY KEY, text TEXT, status TEXT DEFAULT 'queued', created_at REAL, best_score REAL)")
        conn.execute("INSERT INTO requests VALUES ('old','walk backwards','done',0,NULL)")
    conn = db.connect(path)
    assert db.get_request(conn, "old")["training_steps"] is None
