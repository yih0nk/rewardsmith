import importlib
import io
import json
import pickle
import zipfile

import pytest
from fastapi.testclient import TestClient

from rewardsmith import db, export
from rewardsmith.sim import progress
from rewardsmith.sim.fake import FakeSimulator

REWARD = "def reward(state, action, prev_action):\n    return {'spin': state['yaw_rate']}\n"
SCENE = {"objects": [{"name": "ball", "shape": "sphere", "size": [0.1], "pos": [1.0, 0.0, 0.1], "movable": True}],
         "target": "ball"}


def _client(tmp_path, monkeypatch) -> TestClient:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("REWARDSMITH_HOME", str(tmp_path))
    import rewardsmith.server as server

    importlib.reload(server)
    return TestClient(server.app)


def _trained(tmp_path, metrics=None, checkpoints=(100, 200), scene=None) -> int:
    """A scored candidate whose artifacts a fake run wrote, plus params and a video."""
    conn = db.connect(tmp_path / "rewardsmith.db")
    rid = db.create_request(conn, "spin in place", scene)
    db.set_success_spec(conn, rid, [{"metric": "yaw_rate", "op": ">", "value": 1.0}])
    round_id = db.add_round(conn, rid, 1)
    cid = db.add_candidate(conn, round_id, 0, REWARD)
    out_dir = progress.candidate_dir(tmp_path / "artifacts", rid, 1, 0)
    result = FakeSimulator().run(REWARD, out_dir)
    for steps in checkpoints:
        (out_dir / f"params_{steps}.pkl").write_bytes(pickle.dumps({"steps": steps}))
    (out_dir / "rollout.mp4").write_bytes(b"fake mp4")
    db.set_candidate(
        conn, cid, status="scored", score=7.5, critique="Spins cleanly.", idea="reward yaw rate",
        metrics={**result.metrics, **(metrics or {})}, video_path=f"{rid}/r1c0/rollout.mp4",
        check_result={"pass": True, "clauses": []},
    )
    return cid


def _bundle(response) -> tuple[str, zipfile.ZipFile]:
    assert response.status_code == 200, response.text
    assert response.headers["content-type"] == "application/zip"
    z = zipfile.ZipFile(io.BytesIO(response.content))
    root = z.namelist()[0].split("/")[0]
    assert response.headers["content-disposition"] == f'attachment; filename="{root}.zip"'
    return root, z


def test_export_holds_the_policy_its_task_and_a_runner(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    cid = _trained(tmp_path)

    root, z = _bundle(client.get(f"/api/candidates/{cid}/export"))
    names = {n.removeprefix(root + "/") for n in z.namelist()}
    assert {
        "README.md", "run_policy.py", "manifest.json", "policy/params.pkl", "policy/network.json",
        "task/reward.py", "task/scene.json", "task/success_check.json", "task/metrics.json", "task/critique.json",
        "rollout.mp4", "rewardsmith_env/__init__.py", *(f"rewardsmith_env/{f}" for f in export.ENV_FILES),
    } == names  # joint mode: no walker

    read = lambda path: z.read(f"{root}/{path}")  # noqa: E731
    assert pickle.loads(read("policy/params.pkl")) == {"steps": 200}  # the latest checkpoint
    manifest = json.loads(read("manifest.json"))
    assert manifest["mode"] == "joints" and manifest["action_size"] == 12
    assert manifest["checkpoint"] == "params_200.pkl" and manifest["score"] == 7.5
    assert manifest["task"] == "spin in place" and manifest["candidate_id"] == cid
    assert json.loads(read("policy/network.json")) == {**export.NETWORK, "action_size": 12, "mode": "joints"}
    assert read("task/reward.py").decode() == REWARD
    assert json.loads(read("task/scene.json")) is None
    check = json.loads(read("task/success_check.json"))
    assert check["spec"][0]["metric"] == "yaw_rate" and check["result"]["pass"] is True
    assert json.loads(read("task/metrics.json"))["upright"] == 0.98
    assert json.loads(read("task/critique.json"))["critique"] == "Spins cleanly."
    assert read("rollout.mp4") == b"fake mp4"
    assert "spin in place" in read("README.md").decode()
    assert read("rewardsmith_env/go1.py") == (export._SIM_DIR / "go1.py").read_bytes()


def test_steer_export_carries_the_walker_and_the_scene(tmp_path, monkeypatch):
    walker = tmp_path / "walker.pkl"
    walker.write_bytes(pickle.dumps("walker"))
    monkeypatch.setenv("REWARDSMITH_WALKER", str(walker))
    client = _client(tmp_path, monkeypatch)
    cid = _trained(tmp_path, metrics={"steered": 1}, scene=SCENE)

    root, z = _bundle(client.get(f"/api/candidates/{cid}/export"))
    assert pickle.loads(z.read(f"{root}/policy/walker.pkl")) == "walker"
    assert json.loads(z.read(f"{root}/manifest.json"))["action_size"] == 3
    assert json.loads(z.read(f"{root}/policy/network.json"))["mode"] == "steer"
    assert json.loads(z.read(f"{root}/task/scene.json"))["target"] == "ball"


def test_steer_export_without_the_walker_is_refused(tmp_path, monkeypatch):
    monkeypatch.setenv("REWARDSMITH_WALKER", str(tmp_path / "missing.pkl"))
    client = _client(tmp_path, monkeypatch)
    cid = _trained(tmp_path, metrics={"steered": 1})

    r = client.get(f"/api/candidates/{cid}/export")
    assert r.status_code == 409 and "walker" in r.json()["detail"]


def test_untrained_or_unknown_candidates_have_nothing_to_export(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    cid = _trained(tmp_path, checkpoints=())

    assert client.get(f"/api/candidates/{cid}/export").status_code == 409
    assert client.get("/api/candidates/999/export").status_code == 404


def test_a_final_params_file_wins_over_the_checkpoints(tmp_path):
    (tmp_path / "params_300.pkl").write_bytes(b"")
    assert export.policy_file(tmp_path).name == "params_300.pkl"
    (tmp_path / "params.pkl").write_bytes(b"")
    assert export.policy_file(tmp_path).name == "params.pkl"


def test_network_matches_what_training_uses():
    locomotion_params = pytest.importorskip("mujoco_playground.config.locomotion_params")
    from rewardsmith.sim.go1_spec import COMMAND_LIMITS

    factory = dict(locomotion_params.brax_ppo_config("Go1JoystickFlatTerrain").network_factory)
    assert {k: list(v) if isinstance(v, tuple) else v for k, v in factory.items()} == export.NETWORK
    assert len(COMMAND_LIMITS) == export.STEER_ACTIONS
