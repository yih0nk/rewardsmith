import time
from dataclasses import replace
from types import SimpleNamespace

import pytest

from rewardsmith import config, db
from rewardsmith.agents.fake import FakeCritic, FakeEditor, FakeRanker, FakeWriter
from rewardsmith.orchestrator import run_request
from rewardsmith.search import Halving, parse_halving
from rewardsmith.sim.fake import FakeSimulator
from rewardsmith.sim.progress import ProgressWriter


def _evals(out_dir, rewards):
    w = ProgressWriter(out_dir, 1000)
    for n, r in enumerate(rewards):
        w.point(SimpleNamespace(steps=n * 100, wall_s=float(n), episode_reward=r, terms={}, behavior={"forward_velocity": r}))


def _settle(cull, running, timeout=5.0):
    """Poll like run_many does until the cull names someone (or gives up)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        stop = cull(running)
        if stop:
            return stop
        time.sleep(0.01)
    return {}


def test_parse_halving():
    assert parse_halving("") == ()
    assert parse_halving("8") == ()
    assert parse_halving("8, 4,2") == (8, 4, 2)
    with pytest.raises(ValueError):
        parse_halving("4,4")


def test_waits_for_every_runner_then_stops_the_bottom(tmp_path):
    dirs = [tmp_path / f"c{i}" for i in range(4)]
    for d, r in zip(dirs, [1.0, 4.0, 3.0]):
        _evals(d, [0.0, r])
    _evals(dirs[3], [0.0])  # still before its first eval
    seen = []
    cull = Halving(dirs, [2], rank=lambda snaps: seen.append(snaps) or FakeRanker().rank("", snaps))

    assert _settle(cull, {0, 1, 2, 3}, timeout=0.2) == {}  # c3 hasn't reached the rung yet
    _evals(dirs[3], [0.0, 2.0])
    stop = _settle(cull, {0, 1, 2, 3})

    assert set(stop) == {0, 3}  # rewards 1 and 2 lose to 4 and 3
    assert "ranked 3 of 4" in stop[3] and "top 2" in stop[3]
    assert [s["forward_velocity"] for s in seen[0]] == [1.0, 4.0, 3.0, 2.0]
    assert cull.finished


def test_a_failing_ranker_keeps_everyone(tmp_path):
    dirs = [tmp_path / f"c{i}" for i in range(3)]
    for d in dirs:
        _evals(d, [0.0, 1.0])
    events = []

    def broken(snaps):
        raise RuntimeError("no")

    cull = Halving(dirs, [1], rank=broken, on_event=events.append)
    assert _settle(cull, {0, 1, 2}, timeout=0.5) == {}
    assert cull.finished and "keeping all" in events[0]


def test_fake_round_with_halving_trains_eight_and_scores_two(tmp_path):
    s = replace(
        config.Settings(
            db_path=tmp_path / "t.db", artifacts_dir=tmp_path / "a", model="m", env_name="Go1JoystickFlatTerrain",
            candidates_per_round=4, max_rounds=2, success_score=99, fake_sim=True, fake_llm=True,
        ),
        halving=(8, 4, 2),
    )
    conn = db.connect(s.db_path)
    rid = db.create_request(conn, "walk backwards")
    run_request(conn, s, db.claim_next_request(conn), FakeWriter(), FakeEditor(), FakeCritic(), FakeSimulator(), FakeRanker())

    detail = db.get_request(conn, rid)
    for r in detail["rounds"]:
        statuses = sorted(c["status"] for c in r["candidates"])
        assert statuses == ["scored"] * 2 + ["stopped"] * 6
        assert all(c["metrics"] and c["error"].startswith("Stopped at") for c in r["candidates"] if c["status"] == "stopped")
    halving_events = [e["message"] for e in detail["events"] if e["kind"] == "halving"]
    assert len(halving_events) == 4  # two rungs per round
    assert "kept candidates r1c" in halving_events[0]
