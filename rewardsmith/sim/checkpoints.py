"""Policy checkpoints on disk, and fitting one to an env with more observations.

A checkpoint is brax PPO's params tuple (normalizer, policy, value), the same
thing ppo.train returns and takes as restore_params, pickled with every array
as numpy so any process can load it without a GPU.

Training children write params_<steps>.pkl at every eval (see train.train). The
pretrained walker (scripts/train_walker.py) is one of these too.
"""

import os
import pickle
import re
from pathlib import Path
from typing import Any

import numpy as np

PARAMS_GLOB = "params_*.pkl"
_STEPS = re.compile(r"params_(\d+)\.pkl$")


def save(path: Path, params: Any) -> Path:
    """Write params to path atomically (a reader never sees half a file)."""
    import jax  # here, so the orchestrator can use latest() without importing jax

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(jax.tree_util.tree_map(np.asarray, params), f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)
    return path


def load(path: Path) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def step_path(out_dir: Path, steps: int) -> Path:
    return Path(out_dir) / f"params_{int(steps)}.pkl"


def latest(out_dir: Path) -> Path | None:
    """The checkpoint with the most steps in out_dir, or None."""
    found = [(int(m.group(1)), p) for p in Path(out_dir).glob(PARAMS_GLOB) if (m := _STEPS.search(p.name))]
    return max(found)[1] if found else None


def obs_sizes(params: Any) -> dict[str, int]:
    """Observation size per obs key, read from the normalizer."""
    return {k: int(np.shape(v)[-1]) for k, v in params[0].mean.items()}


def fit(params: Any, sizes: dict[str, int], obs_keys: dict[str, str]) -> Any:
    """Fit a checkpoint to an env whose observations append extra inputs at the end.

    Our scene env's observation is the joystick observation plus scene features
    (go1.Go1Env._scene_obs), so a walker checkpoint sees fewer inputs than the
    env gives. For each obs key that grew, the normalizer gets mean 0 and std 1
    for the new inputs, and the first layer of every network reading that key
    gets zero weights for them. The fitted policy then acts exactly as the
    checkpoint did, whatever the scene features are, and training learns how to
    use them. Equal sizes pass through; a smaller env is an error.

    obs_keys maps "policy" / "value" to the obs key that network reads.
    """
    normalizer, policy, value = params
    old = obs_sizes(params)
    extra = {}
    for key, new in sizes.items():
        if key not in old:
            raise ValueError(f"checkpoint has no observation {key!r} (it has {sorted(old)})")
        if new < old[key]:
            raise ValueError(f"checkpoint expects {old[key]} {key!r} inputs; this env gives only {new}")
        extra[key] = new - old[key]
    if not any(extra.values()):
        return params

    def pad(tree: dict, n: int, fill: float) -> dict:
        return {k: (np.concatenate([v, np.full(v.shape[:-1] + (extra[k],), fill, v.dtype)], -1)
                    if extra.get(k) else v) for k, v in tree.items()}

    count = float(np.asarray(_count(normalizer.count)))
    normalizer = normalizer.replace(
        mean=pad(normalizer.mean, 0, 0.0),
        std=pad(normalizer.std, 0, 1.0),
        # std = sqrt(summed_variance / count): keep the new inputs at std 1 as the counts grow
        summed_variance=pad(normalizer.summed_variance, 0, count),
    )
    policy = _pad_first_layer(policy, extra.get(obs_keys["policy"], 0))
    value = _pad_first_layer(value, extra.get(obs_keys["value"], 0))
    return normalizer, policy, value


def _count(count: Any) -> Any:
    """brax keeps the normalizer count as a plain array or a UInt64(hi, lo)."""
    if hasattr(count, "hi"):
        return np.asarray(count.hi, np.float64) * 2.0**32 + np.asarray(count.lo, np.float64)
    return count


def _pad_first_layer(net: dict, n: int) -> dict:
    if not n:
        return net
    layers = dict(net["params"])
    kernel = layers["hidden_0"]["kernel"]
    zeros = np.zeros((n, kernel.shape[1]), kernel.dtype)
    layers["hidden_0"] = {**layers["hidden_0"], "kernel": np.concatenate([kernel, zeros], 0)}
    return {**net, "params": layers}


MIN_STD = 0.001  # brax's NormalTanhDistribution adds this to softplus(scale)


def reset_noise(params: Any, std: float) -> Any:
    """The same policy mean, with every action's exploration noise reset to std.

    A trained walker explores with std ~0.1 (its scale outputs shrank as it learned),
    too little for PPO to find a new behavior. The policy's last layer outputs
    [mean, raw scale] per action, scale = softplus(raw) + MIN_STD; this zeroes the
    raw-scale weights and sets their bias so every state gets std.
    """
    normalizer, policy, value = params
    layers = dict(policy["params"])
    last = sorted(layers, key=lambda k: int(k.rsplit("_", 1)[1]))[-1]
    kernel, bias = np.array(layers[last]["kernel"]), np.array(layers[last]["bias"])
    n = bias.shape[0] // 2
    kernel[:, n:] = 0.0
    bias[n:] = np.log(np.expm1(std - MIN_STD))  # softplus^-1
    layers[last] = {**layers[last], "kernel": kernel, "bias": bias}
    return normalizer, {**policy, "params": layers}, value
