"""Stop MuJoCo Warp from printing solver-iteration warnings every physics step.

mujoco_warp turns on every overflow warning by default, and the Go1 model's
`iterations="1"` hits the solver-iterations limit on purpose, every step. On the
4090 that printed 13 GB of log in 21 minutes and filled the pod's volume.

We clear only the ITERATIONS and LS_ITERATIONS (linesearch) bits, so real overflows
(contacts, constraints) still warn. LS_ITERATIONS alone was ~1.5 MB of log per run. Imported for its side effect by go1.py; harmless when warp isn't used.
"""

from collections.abc import Callable
from typing import Any

try:
    from mujoco.mjx.third_party.mujoco_warp._src.types import OverflowType
    from mujoco.mjx.warp import mujoco_warp as _mjwp
except ImportError:  # no warp in this environment
    _mjwp = None


_SILENCED = int(OverflowType.ITERATIONS) | int(OverflowType.LS_ITERATIONS) if _mjwp is not None else 0


def quiet(put_model: Callable[..., Any]) -> Callable[..., Any]:
    def put_model_quietly(*args: Any, **kwargs: Any) -> Any:
        model = put_model(*args, **kwargs)
        model.opt.warn_overflow = int(model.opt.warn_overflow) & ~_SILENCED
        return model

    put_model_quietly.__wrapped__ = put_model
    return put_model_quietly


if _mjwp is not None and not hasattr(_mjwp.put_model, "__wrapped__"):
    _mjwp.put_model = quiet(_mjwp.put_model)
