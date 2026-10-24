import json
import os
from types import SimpleNamespace

import pytest

from rewardsmith.sim import progress
from rewardsmith.sim.fake import FakeSimulator
from rewardsmith.sim.progress import PROGRESS_FILE, ProgressWriter, read_progress

SOURCE = "def reward(state, action, prev_action):\n    return {'task': 1.0, 'energy': -0.1}\n"


def _point(steps, reward, **terms):
    return SimpleNamespace(steps=steps, wall_s=steps / 1000, episode_reward=reward, terms=terms)


def test_stages_and_curve_are_written_in_order(tmp_path):
    clock = [0.0]
    w = ProgressWriter(tmp_path, total_steps=1000, clock=lambda: clock[0])

    w.stage("compiling")
    data = read_progress(tmp_path)
    assert data == {"stage": "compiling", "steps": 0, "total_steps": 1000, "fraction": 0.0, "wall_s": 0.0, "curve": []}

    clock[0] = 30.0
    w.point(_point(0, 0.1, task=0.1))
    clock[0] = 40.0
    w.point(_point(500, 2.5, task=2.0, energy=-0.5))
    data = read_progress(tmp_path)
    assert (data["stage"], data["steps"], data["fraction"], data["wall_s"]) == ("training", 500, 0.5, 40.0)
    assert data["curve"][-1] == {"steps": 500, "wall_s": 0.5, "episode_reward": 2.5, "terms": {"task": 2.0, "energy": -0.5}}
    assert len(data["curve"]) == 2

    w.point(_point(1200, 3.0))  # brax rounds the step count up past num_timesteps
    assert read_progress(tmp_path)["fraction"] == 1.0

    w.stage("rendering")
    assert read_progress(tmp_path)["stage"] == "rendering"
    assert len(read_progress(tmp_path)["curve"]) == 3  # the curve survives stage changes
    w.stage("done")
    assert read_progress(tmp_path)["stage"] == "done"


def test_unknown_stage_is_refused(tmp_path):
    with pytest.raises(ValueError):
        ProgressWriter(tmp_path, 10).stage("exploding")


def test_write_is_atomic(tmp_path, monkeypatch):
    w = ProgressWriter(tmp_path, 1000)
    w.stage("compiling")

    replaced = []
    real_replace = os.replace
    monkeypatch.setattr(progress.os, "replace", lambda src, dst: (replaced.append((src, dst)), real_replace(src, dst)))
    w.point(_point(100, 1.0))
    [(src, dst)] = replaced
    assert os.path.dirname(src) == str(tmp_path) and src != dst  # temp file in the same dir, then renamed
    assert dst == tmp_path / PROGRESS_FILE

    # A write that dies halfway leaves the previous file whole and no temp file behind.
    def broken_dump(data, f, **kw):
        f.write('{"stage": "trai')
        raise OSError("disk full")

    monkeypatch.setattr(progress.json, "dump", broken_dump)
    w.point(_point(200, 2.0))  # logged, not raised: training must keep going
    assert read_progress(tmp_path)["steps"] == 100
    assert sorted(p.name for p in tmp_path.iterdir()) == [PROGRESS_FILE]


def test_nan_becomes_null(tmp_path):
    w = ProgressWriter(tmp_path, 1000)
    w.point(_point(100, float("nan"), task=float("inf")))
    raw = json.loads((tmp_path / PROGRESS_FILE).read_text())  # strict JSON, no NaN tokens
    assert raw["curve"][0]["episode_reward"] is None
    assert raw["curve"][0]["terms"] == {"task": None}


def test_read_progress_missing_or_garbage(tmp_path):
    assert read_progress(tmp_path) is None
    assert read_progress(tmp_path / "nope") is None
    (tmp_path / PROGRESS_FILE).write_text("{not json")
    assert read_progress(tmp_path) is None
    (tmp_path / PROGRESS_FILE).write_text('{"stage": "training", "wall_s": NaN}')
    assert read_progress(tmp_path) == {"stage": "training", "wall_s": None}


def test_fake_run_writes_a_done_progress_file(tmp_path):
    result = FakeSimulator().run(SOURCE, tmp_path / "r1c0")

    data = read_progress(tmp_path / "r1c0")
    assert data["stage"] == "done"
    assert data["fraction"] == 1.0
    assert data["steps"] == data["total_steps"] > 0
    assert [p["episode_reward"] for p in data["curve"]] == result.reward_curve
    steps = [p["steps"] for p in data["curve"]]
    assert steps == sorted(steps) and len(set(steps)) == len(steps)
    assert data["wall_s"] >= data["curve"][-1]["wall_s"] > 0
    # the last eval's terms match the run's term_<name> metrics
    assert data["curve"][-1]["terms"] == {"task": result.metrics["term_task"], "energy": result.metrics["term_energy"]}


def test_behavior_means_are_kept_when_present(tmp_path):
    w = ProgressWriter(tmp_path, 1000)
    w.point(SimpleNamespace(steps=10, wall_s=1.0, episode_reward=0.5, terms={}, behavior={"forward_velocity": -0.4}))
    assert read_progress(tmp_path)["curve"][-1]["behavior"] == {"forward_velocity": -0.4}
