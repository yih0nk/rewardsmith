import importlib

from fastapi.testclient import TestClient


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
