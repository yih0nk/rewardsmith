import importlib
import threading

from fastapi.testclient import TestClient

from rewardsmith import db


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
