"""Claude-written success checks: a small JAX function over one episode's per-step state.

When the fixed metric vocabulary (checker.METRICS, outcomes.py) can't say whether the
task got done (a pose held, stairs climbed, poles weaved, a ball dodged), the checker
writes

    def check(traj):
        climbed = jnp.max(jnp.where(traj["valid"] > 0, traj["base_ground_z"], 0.0)) > 0.2
        return {"climbed": climbed, "top_height": jnp.max(traj["base_ground_z"])}

traj[key] is the reward's state[key] (go1_spec.features_for(scene)) for every control
step of one episode, stacked: shape (T, *shape). Row 0 is the reset state. traj["valid"]
is 1.0 up to and including the step the episode ended on (a fall) and 0.0 after, traj["t"]
is seconds since the reset, and traj["done"] is 1.0 on the step it ended. The function runs
under jax.jit and vmap over episodes, so it must be pure jnp, like a reward.

Each output is a bool or float scalar per episode. Over the evaluation episodes, a bool
becomes the fraction of episodes where it held and a float its mean; the success spec
names them check:<key>, e.g. {"metric": "check:climbed", "op": ">=", "value": 0.8}.

gate() guards the source the way the reward gate guards rewards, plus an AST check: only
jax, jnp and math, no imports, no dunder access, no I/O builtins. It runs the check jitted
on synthetic episodes (a robot standing still where it starts, and one drifting)
and reports, for each key, whether it holds for the standing robot, so a check that passes
without the task (trivially true) can be caught before any training.
"""

import ast
import math
from dataclasses import dataclass, field
from typing import Any

from .go1_spec import CTRL_DT, features_for
from .scene import EMPTY, SceneSpec

CHECK_FILE = "check.py"
MAX_SOURCE = 6000  # characters
MAX_KEYS = 6
GATE_STEPS = 200  # synthetic episode length, steps (4 s)
EXTRA_KEYS = {"valid": "1.0 while the episode is running, including the step it ended on; 0.0 after",
              "done": "1.0 on the step the episode ended (the robot flipped), else 0.0",
              "t": "seconds since the reset (0.0 for row 0)"}
ALLOWED_IMPORTS = frozenset({"jax", "jax.numpy", "math"})
BANNED_NAMES = frozenset({
    "open", "exec", "eval", "compile", "__import__", "globals", "locals", "vars", "getattr", "setattr",
    "delattr", "input", "breakpoint", "exit", "quit", "help", "memoryview", "object", "type", "super",
    "classmethod", "staticmethod", "property", "print",
})
SAFE_BUILTINS = {name: __builtins__[name] if isinstance(__builtins__, dict) else getattr(__builtins__, name)
                 for name in ("abs", "min", "max", "range", "len", "float", "int", "bool", "dict", "list",
                              "tuple", "zip", "enumerate", "sum", "round", "isinstance", "ValueError")}


class CheckCodeError(ValueError):
    """The check source is unusable. The message is meant for the checker."""


@dataclass
class CheckGate:
    keys: dict[str, str] = field(default_factory=dict)  # key -> "bool" | "float"
    standing: dict[str, float] = field(default_factory=dict)  # each key's value on a robot standing still


def contract(scene: SceneSpec = EMPTY) -> str:
    """What check(traj) reads, for the checker's prompt."""
    lines = ["traj keys (each stacked over the episode's steps, shape (T, ...)):"]
    for f in features_for(scene):
        lines.append(f"- traj[\"{f.name}\"] {('(T,)' if not f.shape else f'(T, {f.shape[0]})')}: {f.meaning}")
    for k, meaning in EXTRA_KEYS.items():
        lines.append(f"- traj[\"{k}\"] (T,): {meaning}")
    return "\n".join(lines)


def validate_source(source: str) -> ast.Module:
    """AST guard: raise CheckCodeError naming what is not allowed."""
    if not isinstance(source, str) or not source.strip():
        raise CheckCodeError("the check code is empty")
    if len(source) > MAX_SOURCE:
        raise CheckCodeError(f"the check code is {len(source)} characters; keep it under {MAX_SOURCE}")
    try:
        tree = ast.parse(source, CHECK_FILE)
    except SyntaxError as exc:
        raise CheckCodeError(f"SyntaxError: {exc}") from exc
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            bad = [a.name for a in node.names if a.name not in ALLOWED_IMPORTS]
            if bad:
                raise CheckCodeError(f"only jax, jax.numpy and math may be imported, not {', '.join(bad)}")
        elif isinstance(node, ast.ImportFrom):
            raise CheckCodeError("use `import jax.numpy as jnp`; from-imports are not allowed")
        elif isinstance(node, ast.Attribute) and node.attr.startswith("_"):
            raise CheckCodeError(f"attribute {node.attr!r} is not allowed (no underscore attributes)")
        elif isinstance(node, ast.Name) and (node.id in BANNED_NAMES or node.id.startswith("__")):
            raise CheckCodeError(f"{node.id!r} is not allowed in a check")
        elif isinstance(node, (ast.Global, ast.Nonlocal, ast.ClassDef, ast.AsyncFunctionDef, ast.Await,
                               ast.Yield, ast.YieldFrom, ast.With, ast.AsyncWith, ast.Try, ast.Raise, ast.Delete)):
            raise CheckCodeError(f"{type(node).__name__} statements are not allowed in a check")
        elif isinstance(node, ast.While):
            raise CheckCodeError("while loops are not allowed; use jnp operations over the time axis")
    if not any(isinstance(n, ast.FunctionDef) and n.name == "check" for n in tree.body):
        raise CheckCodeError("the code must define check(traj)")
    return tree


def load_check(source: str):
    tree = validate_source(source)
    import jax
    import jax.numpy as jnp

    namespace: dict[str, Any] = {"__builtins__": dict(SAFE_BUILTINS), "jax": jax, "jnp": jnp, "math": math,
                                 "__name__": "check_module"}
    # Imports in the source re-bind these names; give the restricted builtins an importer for them only.
    namespace["__builtins__"]["__import__"] = _importer
    exec(compile(tree, CHECK_FILE, "exec"), namespace)
    return namespace["check"]


def _importer(name, globals=None, locals=None, fromlist=(), level=0):  # noqa: A002
    if name not in ALLOWED_IMPORTS or level:
        raise ImportError(f"{name} can't be imported in a check")
    import importlib

    module = importlib.import_module(name)
    return module if fromlist else importlib.import_module(name.split(".")[0])


def synthetic_traj(scene: SceneSpec, steps: int = GATE_STEPS, drift: bool = False, seed: int = 0) -> dict[str, Any]:
    """An episode of the robot standing still where it starts (objects 1 m ahead, as in
    go1.dummy_state), or with drift=True one where every value wanders."""
    import jax.numpy as jnp

    from .go1 import dummy_state

    state = dummy_state(scene)
    traj = {k: jnp.broadcast_to(v, (steps, *jnp.shape(v))) for k, v in state.items()}
    if drift:  # deterministic wander, so the gate compiles no random-number kernels
        t = jnp.arange(steps, dtype=jnp.float32)[:, None]
        traj = {k: v + 0.3 * jnp.sin(t * (0.05 + 0.01 * i) + seed + i).reshape((steps,) + (1,) * (v.ndim - 1))
                for i, (k, v) in enumerate(traj.items())}
    traj["t"] = jnp.arange(steps, dtype=jnp.float32) * CTRL_DT
    traj["time"] = traj["t"]
    traj["valid"] = jnp.ones(steps, dtype=jnp.float32)
    traj["done"] = jnp.zeros(steps, dtype=jnp.float32)
    return traj


def gate(source: str, scene: SceneSpec = EMPTY) -> CheckGate:
    """Validate, load and run the check on synthetic episodes. Raise CheckCodeError for the checker."""
    import jax
    import jax.numpy as jnp
    import numpy as np

    with jax.default_device(jax.devices("cpu")[0]):
        try:
            fn = load_check(source)
        except CheckCodeError:
            raise
        except Exception as exc:
            raise CheckCodeError(f"the code failed to load: {type(exc).__name__}: {exc}") from exc
        batch = jax.tree_util.tree_map(lambda a, b: jnp.stack([a, b]),
                                       synthetic_traj(scene), synthetic_traj(scene, drift=True))
        try:
            out = jax.jit(jax.vmap(fn))(batch)
        except KeyError as exc:
            raise CheckCodeError(f"check() read traj[{exc}], which doesn't exist; use only the listed keys") from exc
        except Exception as exc:
            first = (str(exc).strip().splitlines() or [""])[0][:400]
            raise CheckCodeError(f"check() failed under jax.jit: {type(exc).__name__}: {first}. It must be pure "
                                 "jnp: jnp.where instead of if, no Python loops over traced values") from exc
        if not isinstance(out, dict) or not out:
            raise CheckCodeError("check() must return a non-empty dict of named results")
        if len(out) > MAX_KEYS:
            raise CheckCodeError(f"check() returns {len(out)} results; at most {MAX_KEYS}")
        report = CheckGate()
        for key, value in out.items():
            if not isinstance(key, str) or not key.isidentifier():
                raise CheckCodeError(f"result name {key!r} must be a plain identifier like 'climbed'")
            arr = np.asarray(value)
            if arr.shape != (2,):
                raise CheckCodeError(f"result {key!r} must be one scalar per episode, got shape {arr.shape[1:]}")
            if not np.all(np.isfinite(arr.astype(np.float64))):
                raise CheckCodeError(f"result {key!r} is not finite on a test episode; guard divisions and logs")
            report.keys[key] = "bool" if arr.dtype == np.bool_ else "float"
            report.standing[key] = float(arr[0])
        return report


def evaluate(source: str, traj: dict[str, Any]) -> dict[str, float]:
    """check:<key> metrics from a batch of episodes: traj[key] shaped (episodes, T, ...).
    A bool result becomes the fraction of episodes it held in, a float its mean."""
    import jax
    import numpy as np

    out = jax.device_get(jax.jit(jax.vmap(load_check(source)))(traj))
    metrics = {}
    for key, value in out.items():
        arr = np.asarray(value, dtype=np.float64)
        metrics[f"check:{key}"] = float(np.nanmean(arr)) if arr.size else math.nan
    return metrics
