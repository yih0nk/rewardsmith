"""Train Playground's stock Go1 joystick walker once. Every candidate warm-starts from it.

    flock /workspace/gpu.lock env MUJOCO_GL=egl .venv/bin/python scripts/train_walker.py \
        --out /workspace/pretrained/go1_walker [--steps 200000000]

Playground's own env, reward and PPO config (Go1JoystickFlatTerrain), on
domain-randomized robots, as its Go1 notebook trains it. Writes to --out:
  params.pkl         the final params (checkpoints.save format; go1.WALKER_PARAMS points here)
  params_<steps>.pkl every eval's params, to pick an earlier one if it walks as well
  walker.json        training time, the eval curve, and velocity tracking per command
  walker.mp4         a tour: stand, forward, turn, back, sideways, fast forward, stand
The weights stay on the pod; only this script is in git.
"""

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from mujoco_playground import registry

from rewardsmith.sim import checkpoints, render, train, warp_quiet  # noqa: F401  (warp_quiet: no solver spam)
from rewardsmith.sim.go1 import BASE_ENV, default_impl

# (vx m/s, vy m/s, yaw rate rad/s): tracking is measured on each, held for the whole rollout
COMMANDS = {
    "stand": (0.0, 0.0, 0.0),
    "forward_1.0": (1.0, 0.0, 0.0),
    "forward_1.5": (1.5, 0.0, 0.0),
    "backward_0.8": (-0.8, 0.0, 0.0),
    "left_0.5": (0.0, 0.5, 0.0),
    "turn_left_1.0": (0.0, 0.0, 1.0),
    "forward_turn": (0.8, 0.0, 0.6),
}
TOUR = ["stand", "forward_1.0", "turn_left_1.0", "backward_0.8", "left_0.5", "forward_1.5", "stand"]
TOUR_S = 2.5  # seconds per tour leg
TRACK_ENVS, TRACK_STEPS, SETTLE_STEPS = 64, 500, 100


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("/workspace/pretrained/go1_walker"))
    ap.add_argument("--steps", type=int, default=200_000_000, help="Playground's default for this env")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--impl", default=None)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    config = registry.get_default_config(BASE_ENV)
    config.impl = args.impl or default_impl()
    env = registry.load(BASE_ENV, config=config)

    def show(p: train.CurvePoint) -> None:
        terms = " ".join(f"{k}={v:.2f}" for k, v in sorted(p.terms.items()))
        print(f"{p.steps:>12,} steps {p.wall_s:7.1f}s reward {p.episode_reward:7.2f}  {terms}", flush=True)

    # "final" is Playground's config with only num_timesteps/num_evals set; keep its resets per eval.
    result = train.train(
        env, "final", {"num_timesteps": args.steps, "num_evals": 11}, seed=args.seed, on_progress=show,
        randomize=True, checkpoint_dir=args.out,
    )
    checkpoints.save(args.out / "params.pkl", result.params)
    print(f"trained {result.num_timesteps:,} steps in {result.train_s:.0f}s (compile {result.compile_s:.0f}s)", flush=True)

    t0 = time.perf_counter()
    tracking = track(env, result)
    for name, row in tracking.items():
        print(f"{name:>14}: cmd {row['command']}  got {row['achieved']}  fell {row['fell']:.2f}", flush=True)
    video = tour(env, result, args.out)
    report = {
        "env": BASE_ENV,
        "impl": config.impl,
        "num_timesteps": result.num_timesteps,
        "train_s": round(result.train_s, 1),
        "compile_s": round(result.compile_s, 1),
        "steps_per_s": round(result.steps_per_s),
        "obs_sizes": checkpoints.obs_sizes(result.params),
        "curve": [{"steps": p.steps, "wall_s": round(p.wall_s, 1), "episode_reward": round(p.episode_reward, 3),
                   "terms": {k: round(v, 3) for k, v in p.terms.items()}} for p in result.curve],
        "tracking": tracking,
        "eval_s": round(time.perf_counter() - t0, 1),
        "video": str(video),
    }
    (args.out / "walker.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in ("num_timesteps", "train_s", "compile_s", "steps_per_s")}))


def _hold(state, command: jax.Array):
    """Pin the command: Playground resamples it every few seconds otherwise."""
    state.info["command"] = command
    state.info["steps_until_next_cmd"] = jnp.int32(10**6)
    return state


def _commanded_step(env, policy):
    def step(state, rng, command):
        state = _hold(state, command)
        obs = env._get_obs(state.data, state.info)  # the obs must show the new command too
        action, _ = policy(obs, rng)
        return env.step(state.replace(obs=obs), action)

    return step


def track(env, result) -> dict[str, dict]:
    """Mean velocity reached per held command, after a settling second, over TRACK_ENVS robots."""
    policy = result.make_inference_fn(result.params, deterministic=False)
    step = _commanded_step(env, policy)

    def one(rng, command):
        state = env.reset(rng)

        def body(carry, _):
            state, rng, fell = carry
            rng, key = jax.random.split(rng)
            state = step(state, key, command)
            fell = fell | (state.done > 0)
            out = jnp.concatenate([env.get_local_linvel(state.data)[:2], env.get_gyro(state.data)[2:3]])
            return (state, rng, fell), (out, fell)

        _, (vel, fell) = jax.lax.scan(body, (state, rng, jnp.bool_(False)), length=TRACK_STEPS)
        return vel, fell

    run = jax.jit(jax.vmap(one, in_axes=(0, None)))
    keys = jax.random.split(jax.random.PRNGKey(1), TRACK_ENVS)
    out = {}
    for name, cmd in COMMANDS.items():
        vel, fell = jax.device_get(run(keys, jnp.array(cmd, dtype=jnp.float32)))
        upright = ~np.asarray(fell)[:, SETTLE_STEPS:]
        v = np.asarray(vel)[:, SETTLE_STEPS:]
        mean = (v * upright[..., None]).sum((0, 1)) / max(upright.sum(), 1)
        err = np.sqrt((((v - np.array(cmd)) ** 2) * upright[..., None]).sum((0, 1)) / max(upright.sum(), 1))
        out[name] = {
            "command": list(cmd),
            "achieved": [round(float(x), 3) for x in mean],
            "rms_error": [round(float(x), 3) for x in err],
            "fell": round(float(np.asarray(fell)[:, -1].mean()), 3),
        }
    return out


def tour(env, result, out_dir: Path) -> Path:
    policy = result.make_inference_fn(result.params, deterministic=False)
    step = jax.jit(_commanded_step(env, policy))
    legs = [jnp.array(COMMANDS[name], dtype=jnp.float32) for name in TOUR]
    per_leg = round(TOUR_S / env.dt)
    rng = jax.random.PRNGKey(0)
    state = jax.jit(env.reset)(rng)
    qpos = [np.asarray(state.data.qpos)]
    for command in legs:
        for _ in range(per_leg):
            rng, key = jax.random.split(rng)
            state = step(state, key, command)
            qpos.append(np.asarray(state.data.qpos))
    path = out_dir / "walker.mp4"
    render.render_videos(env.mj_model, np.stack(qpos), env.dt, [(path, "chase")], np.array([0]), 720, 1280)
    return path


if __name__ == "__main__":
    main()
