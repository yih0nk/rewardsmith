"""Robustness score: does the trained policy still do its job on robots it has not seen?

After training, the final policy is rolled out, all at once in one batch, on:
  robots  N_ROBOTS robots from Playground's Go1 domain randomizer, with seeds training never
          used (floor friction, link masses, payload, center of mass, joint friction and offsets)
  seeds   the nominal robot from N_SEEDS other start states
  pushes  the nominal robot, shoved front, back, left and right once mid-rollout

Each rollout gets the same metrics as the nominal rollout in render.py (forward_velocity,
yaw_rate, fell, ...) and passes if `check(metrics, nominal)` holds. The default check is a
behavior proxy that needs no spec: no fall, and the main motion metric keeps the nominal
rollout's sign and at least half its magnitude. To use the request's up-front success spec
instead (agents/checker.py, same metric names), pass
`check=lambda m, _: checker.evaluate(spec, m)["pass"]`.
"""

import dataclasses
import json
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from .base import ROLLOUT_STEPS
from .go1 import Go1Env
from .render import rollout_summary
from .train import TrainResult, randomizer

N_ROBOTS = 10
N_SEEDS = 3
PUSH_DIRS = {"front": (1.0, 0.0), "back": (-1.0, 0.0), "left": (0.0, 1.0), "right": (0.0, -1.0)}
PUSH_DV = 0.75  # m/s the shove adds to the whole robot; training pushes reach about 1 m/s
PUSH_AT_S, PUSH_FOR_S = 2.0, 0.1
STEPS = ROLLOUT_STEPS  # same length as the nominal rollout
SEED = 10_007  # far from training's seeds, so the robots are new

# Below these the robot isn't really moving that way, and there is no motion to keep.
MIN_MOTION = {"forward_velocity": 0.1, "lateral_velocity": 0.1, "yaw_rate": 0.2}
KEEP_FRACTION = 0.5

Metrics = Mapping[str, float]
Check = Callable[[Metrics, Metrics], bool]


def main_motion(nominal: Metrics) -> str | None:
    """The motion metric the behavior is about: the largest relative to its threshold, if any."""
    ratios = {k: abs(float(nominal.get(k, 0.0))) / floor for k, floor in MIN_MOTION.items()}
    best = max(ratios, key=ratios.get)
    return best if ratios[best] >= 1.0 else None


def behavior_proxy(metrics: Metrics, nominal: Metrics) -> bool:
    if metrics["fell"] >= 0.5:
        return False
    key = main_motion(nominal)
    if key is None:
        return True
    ref, got = float(nominal[key]), float(metrics[key])
    return got * ref > 0 and abs(got) >= KEEP_FRACTION * abs(ref)


@dataclasses.dataclass
class Plan:
    """The robustness rollouts: who they are, and the batch inputs that reproduce them."""

    groups: list[str]
    names: list[str]
    keys: jax.Array  # one reset key per rollout
    push_dirs: list[tuple[float, float]]
    rng_models: jax.Array  # randomizes the first groups.count("robots") robots


def plan(n_robots: int = N_ROBOTS, n_seeds: int = N_SEEDS, pushes: bool = True, seed: int = SEED) -> Plan:
    shoves = PUSH_DIRS if pushes else {}
    groups = ["robots"] * n_robots + ["seeds"] * n_seeds + ["pushes"] * len(shoves)
    names = [f"robot{i}" for i in range(n_robots)] + [f"seed{i}" for i in range(n_seeds)] + list(shoves)
    push_dirs = [(0.0, 0.0)] * (n_robots + n_seeds) + list(shoves.values())
    rng_models, rng_reset = jax.random.split(jax.random.PRNGKey(seed))
    return Plan(groups, names, jax.random.split(rng_reset, len(groups)), push_dirs, rng_models)


def evaluate(
    env: Go1Env,
    result: TrainResult,
    nominal: Metrics,
    out_dir: Path | None = None,
    check: Check = behavior_proxy,
    n_robots: int = N_ROBOTS,
    n_seeds: int = N_SEEDS,
    pushes: bool = True,
    steps: int = STEPS,
    seed: int = SEED,
) -> dict[str, float]:
    """Return robustness metrics for the candidate; also write the per-rollout detail to out_dir."""
    start = time.perf_counter()
    p = plan(n_robots, n_seeds, pushes, seed)
    models, in_axes = models_for(env, n_robots, len(p.groups) - n_robots, p.rng_models)
    _, outs = run_batch(env, result, models, in_axes, p.keys, p.push_dirs, steps)
    rows = [summarize(outs, i) for i in range(len(p.groups))]
    return report(p, rows, models, nominal, check, out_dir, start)


def report(
    p: Plan,
    rows: list[dict[str, float]],
    models: Any,
    nominal: Metrics,
    check: Check = behavior_proxy,
    out_dir: Path | None = None,
    start: float | None = None,
) -> dict[str, float]:
    """The robust_* summary of the plan's rollouts (rows, in plan order; models[i] is row i's robot)."""
    summary: dict[str, float] = {}
    for group in ("robots", "seeds", "pushes"):
        members = [r for r, g in zip(rows, p.groups) if g == group]
        if members:
            summary[f"robust_{group}_pass"] = sum(check(r, nominal) for r in members)
            summary[f"robust_{group}_n"] = len(members)
    if start is not None:
        summary["robust_s"] = round(time.perf_counter() - start, 1)

    if out_dir is not None:
        detail = {
            "check": getattr(check, "__name__", "custom"),
            "main_motion": main_motion(nominal),
            "nominal": dict(nominal),
            "rollouts": [
                {"name": name, "group": g, "pass": check(r, nominal), **r, **_physics(models, i)}
                for i, (name, g, r) in enumerate(zip(p.names, p.groups, rows))
            ],
        }
        (out_dir / "robustness.json").write_text(json.dumps(detail, indent=2))
    return summary


def summarize(outs: Mapping[str, np.ndarray], i: int) -> dict[str, float]:
    """Rollout i's metrics, averaged over the steps up to and including its fall (as render.py does)."""
    summary, _ = rollout_summary(outs["qpos"][i], outs["lin"][i], outs["yaw"][i], outs["up"][i], outs["done"][i])
    return summary


def models_for(env: Go1Env, n_random: int, n_nominal: int, rng: jax.Array) -> tuple[Any, Any]:
    """A batch of n_random randomized robots followed by n_nominal copies of the nominal one."""
    nominal = env.mjx_model
    models, in_axes = randomizer()(nominal, jax.random.split(rng, n_random))
    batched = {
        f.name: jnp.concatenate([
            getattr(models, f.name),
            jnp.broadcast_to(getattr(nominal, f.name), (n_nominal, *getattr(nominal, f.name).shape)),
        ])
        for f in dataclasses.fields(in_axes)
        if _is_batched(getattr(in_axes, f.name), getattr(nominal, f.name))
    }
    return models.tree_replace(batched), in_axes


def _is_batched(axis: Any, value: Any) -> bool:
    # in_axes holds 0 for the randomized fields and None for the rest, but the model's static
    # fields (ints, numpy arrays) pass through tree_map unchanged and can be 0 as well.
    return type(axis) is int and axis == 0 and isinstance(value, (jax.Array, np.ndarray))


def run_batch(
    env: Go1Env,
    result: TrainResult,
    models: Any,
    in_axes: Any,
    keys: jax.Array,
    push_dirs: list[tuple[float, float]],
    steps: int = STEPS,
    observe: Callable[..., Any] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Roll the policy out in every env of a batch at once, in one compiled scan.

    Env i runs robot models[i] (in_axes from models_for; None for one shared model) from
    reset key keys[i], with action keys split off the same key each step, and push_dirs[i]
    (zero for no shove). Returns (reset, steps): reset holds qpos, mocap_pos, mocap_quat
    and target_dist per env; steps holds qpos, lin, yaw, up, done and target_dist per env
    and step, shaped (env, step, ...). target_dist only when the scene has a target.
    An optional observer adds nested scene_outcome telemetry at reset and each step.
    """
    rollout = _rollout_fn(env, result, steps, observe=observe)
    reset, outs = jax.device_get(jax.jit(jax.vmap(rollout, in_axes=(in_axes, 0, 0)))(
        models, keys, jnp.array(push_dirs, dtype=jnp.float32)
    ))
    return jax.tree_util.tree_map(np.asarray, (reset, outs))


def _rollout_fn(env: Go1Env, result: TrainResult, steps: int, observe=None) -> Callable[..., Any]:
    policy = result.make_inference_fn(result.params, deterministic=False)
    torso = env._torso_body_id
    push_from = round(PUSH_AT_S / env.dt)
    push_to = push_from + max(1, round(PUSH_FOR_S / env.dt))
    push_force = float(env._torso_mass) * PUSH_DV / ((push_to - push_from) * env.dt)
    has_target = env.scene.target_object is not None

    def rollout(model, rng: jax.Array, push_dir: jax.Array):
        nominal = env._mjx_model
        env._mjx_model = model  # same swap as Playground's domain-randomization wrapper
        try:
            state = env.reset(rng)
            d = state.data
            reset = {"qpos": d.qpos, "mocap_pos": d.mocap_pos, "mocap_quat": d.mocap_quat}
            if observe is not None:
                reset["scene_outcome"] = observe(state)
            if has_target:
                reset["target_dist"] = env.scene_features(d)["target_dist"]

            def step(carry, t):
                state, rng = carry
                rng, key = jax.random.split(rng)
                # The shove is along the robot's own heading axes, flattened to the ground.
                heading = state.data.xmat[torso] @ jnp.array([push_dir[0], push_dir[1], 0.0])
                heading = heading.at[2].set(0.0)
                heading = heading / (jnp.linalg.norm(heading) + 1e-6) * jnp.linalg.norm(push_dir)
                on = (t >= push_from) & (t < push_to)
                xfrc = jnp.zeros_like(state.data.xfrc_applied).at[torso, :3].set(on * push_force * heading)
                action, _ = policy(state.obs, key)
                state = env.step(state.replace(data=state.data.replace(xfrc_applied=xfrc)), action)
                d = state.data
                sample = {
                    "qpos": d.qpos,
                    "lin": env.get_local_linvel(d),
                    "yaw": env.get_gyro(d)[2],
                    "up": env.get_upvector(d)[2],
                    "done": state.done,
                }
                if observe is not None:
                    sample["scene_outcome"] = observe(state)
                if has_target:
                    sample["target_dist"] = env.scene_features(d)["target_dist"]
                return (state, rng), sample

            _, outs = jax.lax.scan(step, (state, rng), jnp.arange(steps))
        finally:
            env._mjx_model = nominal
        return reset, outs

    return rollout


def _physics(models: Any, i: int) -> dict[str, float]:
    """Robot i's floor friction and total mass, for robustness.json."""
    return {
        "floor_friction": round(float(models.geom_friction[i, 0, 0]), 3),
        "mass_kg": round(float(np.asarray(models.body_mass[i]).sum()), 2),
    }
