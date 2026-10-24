import importlib
import os
import sqlite3
import threading

from fastapi.testclient import TestClient

from rewardsmith import config, db


def test_create_and_read_request(tmp_path, monkeypatch):
    monkeypatch.setenv("REWARDSMITH_HOME", str(tmp_path))
    import rewardsmith.server as server

    importlib.reload(server)
    client = TestClient(server.app)

    created = client.post("/api/requests", json={"text": "walk backwards"})
    assert created.status_code == 201
    rid = created.json()["id"]

    listed = client.get("/api/requests").json()
    assert [r["id"] for r in listed] == [rid]

    detail = client.get(f"/api/requests/{rid}").json()
    assert detail["status"] == "queued"
    assert detail["rounds"] == []

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


def _client(tmp_path, monkeypatch) -> TestClient:
    monkeypatch.setenv("REWARDSMITH_HOME", str(tmp_path))
    import rewardsmith.server as server

    importlib.reload(server)
    return TestClient(server.app)


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
