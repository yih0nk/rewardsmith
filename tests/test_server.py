import importlib
import os
import sqlite3
import threading
import time
from types import SimpleNamespace

from fastapi.testclient import TestClient

from rewardsmith import config, db


def _client(tmp_path, monkeypatch) -> TestClient:
    """A fresh server on a db in tmp_path. Runs from tmp_path so the server
    module's .env loading can't pick up a developer's .env."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("REWARDSMITH_HOME", str(tmp_path))
    import rewardsmith.server as server

    importlib.reload(server)
    return TestClient(server.app)


def test_create_and_read_request(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)

    created = client.post("/api/requests", json={"text": "walk backwards"})
    assert created.status_code == 201
    rid = created.json()["id"]

    listed = client.get("/api/requests").json()
    assert [r["id"] for r in listed] == [rid]

    detail = client.get(f"/api/requests/{rid}").json()
    assert detail["status"] == "queued"
    assert detail["rounds"] == []
    assert detail["stages"] == [] and detail["stage"] is None
    assert abs(detail["server_time"] - time.time()) < 60

    assert client.get("/api/requests/missing").status_code == 404
    assert client.post("/api/requests", json={"text": "x"}).status_code == 422


def test_connection_works_across_threads(tmp_path):
    # The API opens a connection in one worker thread and may query it from another.
    box = {}
    t = threading.Thread(target=lambda: box.update(conn=db.connect(tmp_path / "t.db")))
    t.start()
    t.join()
    rid = db.create_request(box["conn"], "walk backwards")
    assert db.get_request(box["conn"], rid)["text"] == "walk backwards"


def _scored_candidate(tmp_path) -> tuple[str, int]:
    conn = db.connect(tmp_path / "rewardsmith.db")
    rid = db.create_request(conn, "spin in place")
    round_id = db.add_round(conn, rid, 1)
    cid = db.add_candidate(conn, round_id, 0, "def reward(state, action, prev_action): ...")
    db.set_candidate(conn, cid, status="scored", score=7.5, metrics={"term_spin": 2.0}, video_path="x/r1c0/rollout.mp4")
    return rid, cid


def test_flag_and_list_hall_of_fame(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    rid, cid = _scored_candidate(tmp_path)
    assert client.get("/api/hall-of-fame").json() == []

    assert client.post(f"/api/candidates/{cid}/flag", json={"note": "  spins by vibrating  "}).status_code == 200
    [entry] = client.get("/api/hall-of-fame").json()
    assert (entry["id"], entry["request_id"], entry["request_text"], entry["round_n"]) == (cid, rid, "spin in place", 1)
    assert entry["flag_note"] == "spins by vibrating"
    assert entry["metrics"] == {"term_spin": 2.0}
    assert entry["video_path"] == "x/r1c0/rollout.mp4"

    # The request detail carries the flag too, so the board can show it.
    detail = client.get(f"/api/requests/{rid}").json()
    assert detail["rounds"][0]["candidates"][0]["flag_note"] == "spins by vibrating"

    assert client.delete(f"/api/candidates/{cid}/flag").status_code == 200
    assert client.get("/api/hall-of-fame").json() == []


def test_non_finite_metrics_are_saved_as_null(tmp_path, monkeypatch):
    # A diverged run can report NaN or inf. json.dumps would store them as NaN/Infinity,
    # which is not JSON; the column must stay valid JSON and the API must still answer.
    client = _client(tmp_path, monkeypatch)
    rid, cid = _scored_candidate(tmp_path)
    conn = db.connect(tmp_path / "rewardsmith.db")
    metrics = {"episode_reward": float("nan"), "term_spin": float("inf"), "yaw_rate": -float("inf"),
               "fell": 0.0, "nested": {"curve": [1.0, float("nan")]}}
    db.set_candidate(conn, cid, metrics=metrics)
    db.flag_candidate(conn, cid, "diverged")

    stored = conn.execute("SELECT metrics FROM candidates WHERE id = ?", (cid,)).fetchone()[0]
    assert "NaN" not in stored and "Infinity" not in stored
    want = {"episode_reward": None, "term_spin": None, "yaw_rate": None, "fell": 0.0, "nested": {"curve": [1.0, None]}}
    detail = client.get(f"/api/requests/{rid}")
    assert detail.status_code == 200
    assert detail.json()["rounds"][0]["candidates"][0]["metrics"] == want
    hall = client.get("/api/hall-of-fame")
    assert hall.status_code == 200
    assert hall.json()[0]["metrics"] == want


def test_nan_saved_by_an_older_version_reads_as_null(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    rid, cid = _scored_candidate(tmp_path)
    conn = db.connect(tmp_path / "rewardsmith.db")
    conn.execute("UPDATE candidates SET metrics = ? WHERE id = ?", ('{"episode_reward": NaN, "fell": 0.0}', cid))

    detail = client.get(f"/api/requests/{rid}")
    assert detail.status_code == 200
    assert detail.json()["rounds"][0]["candidates"][0]["metrics"] == {"episode_reward": None, "fell": 0.0}


def test_flag_rejects_bad_input(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    _, cid = _scored_candidate(tmp_path)
    assert client.post("/api/candidates/999/flag", json={"note": "x"}).status_code == 404
    assert client.delete("/api/candidates/999/flag").status_code == 404
    assert client.post(f"/api/candidates/{cid}/flag", json={"note": "   "}).status_code == 422
    assert client.post(f"/api/candidates/{cid}/flag", json={"note": "x" * 201}).status_code == 422


def test_old_db_file_gains_new_columns(tmp_path):
    path = tmp_path / "old.db"
    old = sqlite3.connect(path)
    old.executescript(db.SCHEMA.replace("flagged_at   REAL,", "").replace("flag_note    TEXT", "unused TEXT"))
    old.close()
    conn = db.connect(path)
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(candidates)")}
    assert {"flagged_at", "flag_note"} <= cols


def test_fail_interrupted_requests(tmp_path):
    conn = db.connect(tmp_path / "t.db")
    stuck = db.create_request(conn, "walk backwards")
    waiting = db.create_request(conn, "spin")
    db.set_request(conn, stuck, status="running")

    assert db.fail_interrupted(conn) == [stuck]
    assert db.get_request(conn, stuck)["status"] == "failed"
    assert "restarted" in db.get_request(conn, stuck)["events"][-1]["message"]
    assert db.get_request(conn, waiting)["status"] == "queued"


def test_dotenv_fills_unset_variables_only(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("# comment\nRS_TEST_A=from-file\nRS_TEST_B='quoted'\nRS_TEST_EMPTY=\nRS_TEST_SET=from-file\n")
    for key in ("RS_TEST_A", "RS_TEST_B", "RS_TEST_EMPTY"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("RS_TEST_SET", "from-shell")

    config.load_dotenv(env)
    assert os.environ["RS_TEST_A"] == "from-file"
    assert os.environ["RS_TEST_B"] == "quoted"
    assert "RS_TEST_EMPTY" not in os.environ
    assert os.environ["RS_TEST_SET"] == "from-shell"
    for key in ("RS_TEST_A", "RS_TEST_B"):
        monkeypatch.delenv(key)


def test_request_text_is_trimmed_before_validation(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    assert client.post("/api/requests", json={"text": "     "}).status_code == 422
    assert client.post("/api/requests", json={"text": "  ab  "}).status_code == 422
    rid = client.post("/api/requests", json={"text": "  spin  "}).json()["id"]
    assert client.get(f"/api/requests/{rid}").json()["text"] == "spin"


def test_settings_ignore_dotenv_until_an_entry_point_loads_it(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("REWARDSMITH_ROUNDS=42\n")
    monkeypatch.delenv("REWARDSMITH_ROUNDS", raising=False)
    assert config.load().max_rounds == 5
    config.load_dotenv()
    assert config.load().max_rounds == 42
    monkeypatch.delenv("REWARDSMITH_ROUNDS")


def test_request_detail_attaches_live_progress(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    conn = db.connect(tmp_path / "rewardsmith.db")
    rid = db.create_request(conn, "walk backwards")
    round_id = db.add_round(conn, rid, 1)
    training = db.add_candidate(conn, round_id, 0, "def reward(state, action, prev_action): ...")
    waiting = db.add_candidate(conn, round_id, 1, "def reward(state, action, prev_action): ...")
    db.set_candidate(conn, training, status="training")

    # The orchestrator's folder convention: artifacts/<rid>/r<round n>c<idx>/
    from rewardsmith.sim.progress import ProgressWriter

    w = ProgressWriter(tmp_path / "artifacts" / rid / "r1c0", total_steps=1000)
    w.point(SimpleNamespace(steps=250, wall_s=31.0, episode_reward=1.5, terms={"task": 1.2}))

    cands = client.get(f"/api/requests/{rid}").json()["rounds"][0]["candidates"]
    by_id = {c["id"]: c for c in cands}
    p = by_id[training]["progress"]
    assert (p["stage"], p["steps"], p["total_steps"], p["fraction"]) == ("training", 250, 1000, 0.25)
    assert p["curve"] == [{"steps": 250, "wall_s": 31.0, "episode_reward": 1.5, "terms": {"task": 1.2}}]
    assert by_id[waiting]["progress"] is None


def test_request_with_a_scene(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    ball = {"name": "ball", "kind": "ball", "color": "red", "size": [0.15], "distance": [1.5, 3]}

    rid = client.post("/api/requests", json={"text": "walk to the ball", "scene": {"objects": [ball]}}).json()["id"]
    scene = client.get(f"/api/requests/{rid}").json()["scene"]
    assert scene["target"] == "ball"
    assert scene["objects"][0]["bearing"] == [0.0, 0.0]  # stored normalized

    plain = client.post("/api/requests", json={"text": "walk backwards"}).json()["id"]
    assert client.get(f"/api/requests/{plain}").json()["scene"] is None  # the scene writer decides

    bad = client.post("/api/requests", json={"text": "walk to the ball", "scene": {"objects": [{**ball, "size": [9]}]}})
    assert bad.status_code == 422
    assert "radius must be between" in bad.json()["detail"]


def test_old_db_gets_the_scene_column(tmp_path):
    path = tmp_path / "old.db"
    old = sqlite3.connect(path)
    old.execute("CREATE TABLE requests (id TEXT PRIMARY KEY, text TEXT NOT NULL, "
                "status TEXT NOT NULL DEFAULT 'queued', best_score REAL, created_at REAL NOT NULL)")
    old.commit()
    old.close()
    conn = db.connect(path)
    rid = db.create_request(conn, "walk to the ball", {"terrain": "flat", "objects": [], "target": None})
    assert db.get_request(conn, rid)["scene"]["terrain"] == "flat"


def test_request_detail_reports_robustness(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    rid, cid = _scored_candidate(tmp_path)
    [cand] = client.get(f"/api/requests/{rid}").json()["rounds"][0]["candidates"]
    assert cand["robustness"] is None  # not scored for robustness

    conn = db.connect(tmp_path / "rewardsmith.db")
    metrics = {"robust_robots_pass": 8, "robust_robots_n": 10, "robust_pushes_pass": 3, "robust_pushes_n": 4}
    db.set_candidate(conn, cid, metrics=metrics)
    [cand] = client.get(f"/api/requests/{rid}").json()["rounds"][0]["candidates"]
    assert cand["robustness"] == {"robots": {"passed": 8, "total": 10}, "pushes": {"passed": 3, "total": 4}}


def test_malformed_scene_color_returns_validation_error(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    response = client.post("/api/requests", json={"text": "walk to the ball", "scene": {
        "objects": [{"name": "ball", "kind": "ball", "size": [0.15], "distance": 2, "color": []}]}})
    assert response.status_code == 422
    assert "color must be" in response.json()["detail"]


def test_pause_resume_cancel_endpoints(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    rid = client.post("/api/requests", json={"text": "walk backwards"}).json()["id"]

    assert client.post(f"/api/requests/{rid}/pause").json() == {"state": "paused"}
    assert client.get(f"/api/requests/{rid}").json()["status"] == "paused"
    again = client.post(f"/api/requests/{rid}/pause")
    assert again.status_code == 409 and "paused" in again.json()["detail"]
    assert client.post(f"/api/requests/{rid}/resume").json() == {"state": "queued"}
    assert client.post(f"/api/requests/{rid}/cancel").json() == {"state": "cancelled"}
    assert client.post(f"/api/requests/{rid}/resume").status_code == 409

    assert client.post("/api/requests/missing/cancel").status_code == 404
    assert client.post(f"/api/requests/{rid}/explode").status_code == 404
