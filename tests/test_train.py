from pathlib import Path

import pytest

pytest.importorskip("mujoco_playground")

from brax.training import acting  # noqa: E402

from rewardsmith.sim import go1, train  # noqa: E402

WALK_BACKWARDS = (Path(__file__).parent.parent / "examples" / "rewards" / "walk_backwards.py").read_text()


def test_the_evaluator_compiles_once(monkeypatch):
    """brax used to recompile the 1000-step eval rollout after the first eval (train._unpmap_plain)."""
    made = []

    class Recording(acting.Evaluator):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            made.append(self)

    monkeypatch.setattr(acting, "Evaluator", Recording)
    result = train.train(go1.make_env(WALK_BACKWARDS, impl="jax"), "smoke")

    assert len(result.curve) == 3  # the initial eval plus two more
    [evaluator] = made
    assert evaluator._generate_eval_unroll._cache_size() == 1
    assert train.ppo._unpmap is train._BRAX_UNPMAP  # brax is left as we found it
