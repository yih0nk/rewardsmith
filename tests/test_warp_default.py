from types import SimpleNamespace

import pytest

pytest.importorskip("mujoco_playground")

from mujoco.mjx.third_party.mujoco_warp._src.types import OverflowType  # noqa: E402
from mujoco.mjx.warp import mujoco_warp  # noqa: E402

from rewardsmith.sim import go1, warp_quiet  # noqa: E402


def test_quiet_clears_only_the_solver_iterations_warning():
    put_model = warp_quiet.quiet(lambda: SimpleNamespace(opt=SimpleNamespace(warn_overflow=int(OverflowType.ALL))))
    flags = put_model().opt.warn_overflow
    assert not flags & OverflowType.ITERATIONS
    assert flags & OverflowType.NEFC  # real overflows still warn


def test_warp_put_model_is_patched_once():
    assert hasattr(mujoco_warp.put_model, "__wrapped__")
    assert not hasattr(mujoco_warp.put_model.__wrapped__, "__wrapped__")


def test_default_impl(monkeypatch):
    monkeypatch.setenv("REWARDSMITH_IMPL", "jax")
    assert go1.default_impl() == "jax"
    monkeypatch.delenv("REWARDSMITH_IMPL")
    expected = "warp" if go1.jax.default_backend() == "gpu" else "jax"
    assert go1.default_impl() == expected
