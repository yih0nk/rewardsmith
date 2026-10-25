"""Train a Playground stock joystick walker once. Every candidate warm-starts from it.

    /workspace/gpu_job.sh env MUJOCO_GL=egl .venv/bin/python scripts/train_walker.py \
        --out /workspace/pretrained/go1_walker [--steps 200000000] [--env g1]

Playground's own env, reward and PPO config (Go1JoystickFlatTerrain, or the --env
robot's), on domain-randomized robots, as its notebooks train them. Writes to --out:
  params.pkl         the final params (checkpoints.save format; go1.WALKER_PARAMS points here)
  params_<steps>.pkl every eval's params, to pick an earlier one if it walks as well
  walker.json        training time, the eval curve, and velocity tracking per command
  walker.mp4         a tour: stand, forward, turn, back, sideways, fast forward, stand
--eval-only PARAMS skips training and measures (and films) those params.
The weights stay on the pod; only this script is in git.
"""

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np
from mujoco_playground import registry

from rewardsmith.sim import checkpoints, render, train, warp_quiet  # noqa: F401  (warp_quiet: no solver spam)
from rewardsmith.sim.go1 import BASE_ENV, default_impl


@dataclass(frozen=True)
class Walker:
    """How to command and measure one robot's stock joystick env. WALKERS keys are body names."""

    base_env: str
    # (vx m/s, vy m/s, yaw rate rad/s): tracking is measured on each, held for the whole rollout
    commands: dict[str, tuple[float, float, float]]
    tour: list[str]
    hold: Callable[[Any, jax.Array], Any]  # pin a command on a state (Playground resamples it)
    obs: Callable[[Any, Any], Any]  # (env, state) -> the obs for the state's (new) command
    velocity: Callable[[Any, Any], jax.Array]  # (env, data) -> (vx, vy, yaw rate) in the base frame


def _go1_hold(state, command: jax.Array):
    state.info["command"] = command
    state.info["steps_until_next_cmd"] = jnp.int32(10**6)
    return state


def _g1_hold(state, command: jax.Array):
    state.info["command"] = command
    state.info["step"] = jnp.zeros_like(state.info["step"])  # it resamples after 500 steps
    return state


def _g1_obs(env, state):
    contact = jnp.array([state.data.sensordata[env.mj_model.sensor_adr[s]] > 0 for s in env._feet_floor_found_sensor])
    return env._get_obs(state.data, state.info, contact)


WALKERS = {
    "go1": Walker(
        base_env=BASE_ENV,
        commands={
            "stand": (0.0, 0.0, 0.0),
            "forward_1.0": (1.0, 0.0, 0.0),
            "forward_1.5": (1.5, 0.0, 0.0),
            "backward_0.8": (-0.8, 0.0, 0.0),
            "left_0.5": (0.0, 0.5, 0.0),
            "turn_left_1.0": (0.0, 0.0, 1.0),
            "forward_turn": (0.8, 0.0, 0.6),
        },
        tour=["stand", "forward_1.0", "turn_left_1.0", "backward_0.8", "left_0.5", "forward_1.5", "stand"],
        hold=_go1_hold,
        obs=lambda env, state: env._get_obs(state.data, state.info),
        velocity=lambda env, data: jnp.concatenate([env.get_local_linvel(data)[:2], env.get_gyro(data)[2:3]]),
    ),
    # Playground's G1 commands: forward [-1, 1] m/s, sideways [-0.5, 0.5], turn [-1, 1] rad/s.
    "g1": Walker(
        base_env="G1JoystickFlatTerrain",
        commands={
            "stand": (0.0, 0.0, 0.0),
            "forward_0.5": (0.5, 0.0, 0.0),
            "forward_1.0": (1.0, 0.0, 0.0),
            "backward_0.5": (-0.5, 0.0, 0.0),
            "left_0.3": (0.0, 0.3, 0.0),
            "turn_left_0.8": (0.0, 0.0, 0.8),
            "forward_turn": (0.5, 0.0, 0.5),
        },
        tour=["stand", "forward_0.5", "turn_left_0.8", "backward_0.5", "left_0.3", "forward_1.0", "stand"],
        hold=_g1_hold,
        obs=_g1_obs,
        velocity=lambda env, data: jnp.concatenate(
            [env.get_local_linvel(data, "pelvis")[:2], env.get_gyro(data, "pelvis")[2:3]]
        ),
    ),
}
TOUR_S = 2.5  # seconds per tour leg
TRACK_ENVS, TRACK_STEPS, SETTLE_STEPS = 64, 500, 100


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("/workspace/pretrained/go1_walker"))
    ap.add_argument("--steps", type=int, default=200_000_000, help="Playground's default for this env")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--impl", default=None)
    ap.add_argument("--env", choices=sorted(WALKERS), default="go1")
    ap.add_argument("--evals", type=int, default=11)
    ap.add_argument("--eval-only", type=Path, default=None, help="measure these params instead of training")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    walker = WALKERS[args.env]

    config = registry.get_default_config(walker.base_env)
    config.impl = args.impl or default_impl()
    env = registry.load(walker.base_env, config=config)
    if args.eval_only:
        params = jax.tree_util.tree_map(jnp.asarray, checkpoints.load(args.eval_only))
        result = train.TrainResult(make_inference_fn=_inference_fn(env, walker, params), params=params)
        evaluate(env, walker, result, args.out)
        return

    def show(p: train.CurvePoint) -> None:
        terms = " ".join(f"{k}={v:.2f}" for k, v in sorted(p.terms.items()))
        print(f"{p.steps:>12,} steps {p.wall_s:7.1f}s reward {p.episode_reward:7.2f}  {terms}", flush=True)

    # "final" is Playground's config with only num_timesteps/num_evals set; keep its resets per eval.
    result = train.train(
        env, "final", {"num_timesteps": args.steps, "num_evals": args.evals}, seed=args.seed, on_progress=show,
        randomize=True, checkpoint_dir=args.out, body=args.env,  # the body's PPO config and randomizer
    )
    checkpoints.save(args.out / "params.pkl", result.params)
    print(f"trained {result.num_timesteps:,} steps in {result.train_s:.0f}s (compile {result.compile_s:.0f}s)", flush=True)

    report = {
        "env": walker.base_env,
        "impl": config.impl,
        "num_timesteps": result.num_timesteps,
        "train_s": round(result.train_s, 1),
        "compile_s": round(result.compile_s, 1),
        "steps_per_s": round(result.steps_per_s),
        "obs_sizes": checkpoints.obs_sizes(result.params),
        "curve": [{"steps": p.steps, "wall_s": round(p.wall_s, 1), "episode_reward": round(p.episode_reward, 3),
                   "terms": {k: round(v, 3) for k, v in p.terms.items()}} for p in result.curve],
    }
    (args.out / "walker.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in ("num_timesteps", "train_s", "compile_s", "steps_per_s")}), flush=True)
    evaluate(env, walker, result, args.out)


def _inference_fn(env, walker: Walker, params):
    """make_inference_fn for saved params, with the env's tuned network sizes."""
    from brax.training.acme import running_statistics
    from brax.training.agents.ppo import networks as ppo_networks
    from mujoco_playground.config import locomotion_params

    sizes = {k: (n,) for k, n in checkpoints.obs_sizes(params).items()}
    nets = ppo_networks.make_ppo_networks(
        sizes, env.action_size, preprocess_observations_fn=running_statistics.normalize,
        **dict(locomotion_params.brax_ppo_config(walker.base_env).network_factory),
    )
    return ppo_networks.make_inference_fn(nets)


def evaluate(env, walker: Walker, result, out: Path) -> None:
    """Velocity tracking per held command and a tour video, added to walker.json."""
    t0 = time.perf_counter()
    tracking = track(env, walker, result)
    for name, row in tracking.items():
        print(f"{name:>14}: cmd {row['command']}  got {row['achieved']}  fell {row['fell']:.2f}", flush=True)
    video = tour(env, walker, result, out)
    path = out / "walker.json"
    report = json.loads(path.read_text()) if path.exists() else {"env": walker.base_env}
    report.update(tracking=tracking, eval_s=round(time.perf_counter() - t0, 1), video=str(video))
    path.write_text(json.dumps(report, indent=2))


def _commanded_step(env, walker: Walker, policy):
    def step(state, rng, command):
        state = walker.hold(state, command)
        obs = walker.obs(env, state)  # the obs must show the new command too
        action, _ = policy(obs, rng)
        return env.step(state.replace(obs=obs), action)

    return step


def track(env, walker: Walker, result) -> dict[str, dict]:
    """Mean velocity reached per held command, after a settling second, over TRACK_ENVS robots."""
    policy = result.make_inference_fn(result.params, deterministic=False)
    step = _commanded_step(env, walker, policy)

    def one(rng, command):
        state = env.reset(rng)

        def body(carry, _):
            state, rng, fell = carry
            rng, key = jax.random.split(rng)
            state = step(state, key, command)
            fell = fell | (state.done > 0)
            out = walker.velocity(env, state.data)
            return (state, rng, fell), (out, fell)

        _, (vel, fell) = jax.lax.scan(body, (state, rng, jnp.bool_(False)), length=TRACK_STEPS)
        return vel, fell

    run = jax.jit(jax.vmap(one, in_axes=(0, None)))
    keys = jax.random.split(jax.random.PRNGKey(1), TRACK_ENVS)
    out = {}
    for name, cmd in walker.commands.items():
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


def tour(env, walker: Walker, result, out_dir: Path) -> Path:
    policy = result.make_inference_fn(result.params, deterministic=False)
    step = jax.jit(_commanded_step(env, walker, policy))
    legs = [jnp.array(walker.commands[name], dtype=jnp.float32) for name in walker.tour]
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
