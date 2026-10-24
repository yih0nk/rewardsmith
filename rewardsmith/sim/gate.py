"""Compile gate: smoke-test a generated Go1 reward before spending GPU time on it.

Builds the env, JIT-compiles a short vmapped random-action rollout, and checks
that every reward term stays finite. Failures come back as short messages the
reward writer can act on. Only genuine infrastructure bugs raise.
"""

import time
import traceback
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from . import go1
from .scene import SceneSpec

MAX_MESSAGE = 1500
REWARD_FILE = "reward.py"  # filename reward_slot.load_reward compiles the module under

TRACED_IF_HINT = (
    "reward() uses a Python `if`/`and`/`or`/`bool()` on a traced value, which cannot work "
    "inside jax.jit. Use jnp.where(cond, a, b) (or multiply by cond.astype(jnp.float32)) instead."
)
CONCRETE_HINT = (
    "reward() needs a concrete Python number from a traced value (e.g. float(x), math.* on an "
    "array, or a value-dependent shape). Use jnp functions (jnp.exp, jnp.sqrt, ...) and keep "
    "shapes fixed."
)
INT_HINT = (
    "reward() uses a traced value as a Python int (e.g. range(x), list index x, or int(x)). "
    "Index with constants, or use jnp.take / jnp.where."
)
NUMPY_HINT = (
    "reward() passes a traced value to numpy (np.*) or a Python list conversion. Use jnp.* "
    "instead of np.*; everything must stay a JAX array."
)


@dataclass(frozen=True)
class GateReport:
    error: str | None  # None = pass; otherwise a message for the reward writer
    compile_s: float  # trace + lower + XLA compile of the rollout
    run_s: float  # executing the compiled rollout


def check_reward(
    source: str,
    *,
    impl: str | None = None,
    num_envs: int = 4,
    steps: int = 50,
    seed: int = 0,
    scene: SceneSpec | None = None,
) -> str | None:
    """Return None if the reward survives a short jitted rollout, else a message for the writer."""
    return run_gate(source, impl=impl, num_envs=num_envs, steps=steps, seed=seed, scene=scene).error


def run_gate(
    source: str,
    *,
    impl: str | None = None,
    num_envs: int = 4,
    steps: int = 50,
    seed: int = 0,
    scene: SceneSpec | None = None,
) -> GateReport:
    try:
        env = go1.make_env(source, impl=impl, scene=scene)
    except go1.RewardError as exc:
        where = _reward_lines(exc, source)  # make_env chains the original error; point at its line
        return GateReport(_clip(f"{exc}\n{where}" if where else str(exc)), 0.0, 0.0)

    rollout = _make_rollout(env, num_envs, steps)
    rng = jax.random.PRNGKey(seed)

    t0 = time.perf_counter()
    try:
        compiled = jax.jit(rollout).lower(rng).compile()
    except Exception as exc:
        if not _is_reward_fault(exc):
            raise
        return GateReport(_clip(describe_exception(exc, source)), time.perf_counter() - t0, 0.0)
    compile_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    try:
        totals, terms, sim_ok = jax.block_until_ready(compiled(rng))
    except Exception as exc:  # runtime failures are rare; treat like trace errors
        if not _is_reward_fault(exc):
            raise
        return GateReport(_clip(describe_exception(exc, source)), compile_s, time.perf_counter() - t0)
    run_s = time.perf_counter() - t0

    error = _check_finite(np.asarray(totals), {k: np.asarray(v) for k, v in terms.items()}, np.asarray(sim_ok))
    return GateReport(_clip(error) if error else None, compile_s, run_s)


def _make_rollout(env: go1.Go1Env, num_envs: int, steps: int):
    term_names = env.term_names
    reset = jax.vmap(env.reset)
    step = jax.vmap(env.step)

    def rollout(rng):
        reset_rng, act_rng = jax.random.split(rng)
        state = reset(jax.random.split(reset_rng, num_envs))

        def body(state, key):
            action = jax.random.uniform(key, (num_envs, env.action_size), minval=-1.0, maxval=1.0)
            state = step(state, action)
            terms = {k: state.metrics[f"reward/{k}"] for k in term_names}
            sim_ok = jnp.all(jnp.isfinite(state.data.qpos), axis=-1) & jnp.all(jnp.isfinite(state.data.qvel), axis=-1)
            return state, (state.reward, terms, sim_ok)

        _, out = jax.lax.scan(body, state, jax.random.split(act_rng, steps))
        return out  # arrays shaped (steps, num_envs)

    return rollout


def _check_finite(totals: np.ndarray, terms: dict[str, np.ndarray], sim_ok: np.ndarray) -> str | None:
    """Arrays are (steps, num_envs). Return a message naming the first bad term, or None."""
    bad = []
    for name, values in terms.items():
        idx = np.argwhere(~np.isfinite(values))
        if len(idx):
            t, e = idx[0]
            bad.append((int(t), name, int(e), float(values[t, e])))
    if not bad:
        idx = np.argwhere(~np.isfinite(totals))
        if not len(idx):
            return None
        t, e = idx[0]
        return f"the summed reward became {_kind(float(totals[t, e]))} at step {t} (env {e}), though every term was finite."

    bad.sort()
    t, name, e, value = bad[0]
    others = sorted({n for _, n, _, _ in bad} - {name})
    msg = (
        f"reward term '{name}' became {_kind(value)} at step {t} of a random-action rollout (env {e}). "
        "Check for log/sqrt of values that can be <= 0, division by values that can be 0, "
        "or exp/power overflow; guard with jnp.maximum(x, eps), jnp.clip, or a + eps in denominators."
    )
    if others:
        msg += f" Other non-finite terms: {', '.join(repr(n) for n in others)}."
    if not bool(sim_ok[t, e]):
        msg += " Note: the simulation state was also non-finite at that step."
    return msg


def _kind(value: float) -> str:
    return "NaN" if np.isnan(value) else ("+inf" if value > 0 else "-inf")


def _chain(exc: BaseException):
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        yield exc
        exc = exc.__cause__ or exc.__context__


def _frames(exc: BaseException) -> list[traceback.FrameSummary]:
    out = []
    for e in _chain(exc):
        out.extend(traceback.extract_tb(e.__traceback__))
    return out


def _is_reward_fault(exc: BaseException) -> bool:
    """A failure is the reward's if it is a JAX tracer error or it passed through reward code."""
    if isinstance(exc, (jax.errors.ConcretizationTypeError, jax.errors.TracerArrayConversionError,
                        jax.errors.TracerIntegerConversionError)):
        return True
    return any(f.filename == REWARD_FILE or f.name == "_get_reward" for f in _frames(exc))


def describe_exception(exc: BaseException, source: str) -> str:
    """Short, actionable text for a failure while tracing/running the reward."""
    if isinstance(exc, jax.errors.TracerBoolConversionError):
        head = TRACED_IF_HINT
    elif isinstance(exc, jax.errors.TracerIntegerConversionError):
        head = INT_HINT
    elif isinstance(exc, jax.errors.TracerArrayConversionError):
        head = NUMPY_HINT
    elif isinstance(exc, jax.errors.ConcretizationTypeError):
        head = CONCRETE_HINT
    else:
        first = str(exc).strip().splitlines()[0] if str(exc).strip() else ""
        head = f"{type(exc).__name__}: {first[:600]}"
        if "shape" in first.lower():
            head += " (all arrays in an expression must have matching or broadcastable shapes; see state shapes in the spec)"
    where = _reward_lines(exc, source)
    return f"{head}\n{where}" if where else head


def _reward_lines(exc: BaseException, source: str, limit: int = 3) -> str:
    lines = source.splitlines()
    picked, seen = [], set()
    for f in _frames(exc):
        if f.filename != REWARD_FILE or (f.lineno, f.name) in seen:
            continue
        seen.add((f.lineno, f.name))
        text = lines[f.lineno - 1].strip() if f.lineno and 0 < f.lineno <= len(lines) else ""
        picked.append(f'  reward.py, line {f.lineno}, in {f.name}: {text}')
    return ("at:\n" + "\n".join(picked[-limit:])) if picked else ""


def _clip(msg: str) -> str:
    return msg if len(msg) <= MAX_MESSAGE else msg[: MAX_MESSAGE - 3] + "..."
