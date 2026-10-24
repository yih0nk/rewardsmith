"""Train one reward file and print what happened. The fastest way to test a reward by hand.

    .venv/bin/python scripts/train_reward.py examples/rewards/walk_backwards.py --preset smoke
    .venv/bin/python scripts/train_reward.py my_reward.py --preset round --video out.mp4
    .venv/bin/python scripts/train_reward.py examples/rewards/walk_to_ball.py \
        --scene examples/scenes/red_ball.json --preset round --video ball.mp4

--video does a quick rollout and render for checking by eye (D owns the real render path).
"""

import argparse
import json
import time
from pathlib import Path

from rewardsmith.sim import go1, scene, scene_eval, train


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("reward", type=Path)
    ap.add_argument("--preset", default="smoke", choices=sorted(train.PRESETS))
    ap.add_argument("--impl", help="jax or warp (default: warp on a CUDA GPU, jax elsewhere)")
    ap.add_argument("--timesteps", type=int, help="override num_timesteps")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--randomize", action="store_true", help="train on domain-randomized robots")
    ap.add_argument("--push", action="store_true", help="train with random pushes")
    ap.add_argument("--video", type=Path, help="write a rollout mp4 here")
    ap.add_argument("--scene", type=Path, help="scene spec JSON (default: plain flat floor)")
    args = ap.parse_args()

    spec = scene.parse(args.scene.read_text()) if args.scene else None
    t0 = time.perf_counter()
    env = go1.make_env(
        args.reward.read_text(), impl=args.impl, scene=spec,
        overrides=go1.PUSH_OVERRIDES if args.push else None,
    )
    print(f"env built in {time.perf_counter() - t0:.1f}s ({env._config.impl}), terms: {', '.join(env.term_names)}")

    overrides = {"num_timesteps": args.timesteps} if args.timesteps else None

    def show(p: train.CurvePoint) -> None:
        terms = "  ".join(f"{k}={v:.2f}" for k, v in sorted(p.terms.items()))
        print(f"  {p.steps:>10,} steps  {p.wall_s:7.1f}s  reward {p.episode_reward:8.2f}  {terms}")

    result = train.train(env, args.preset, overrides, seed=args.seed, on_progress=show, randomize=args.randomize)
    print(json.dumps({
        "compile_s": round(result.compile_s, 1),
        "train_s": round(result.train_s, 1),
        "steps_per_s": round(result.steps_per_s),
        "final_reward": result.reward_curve[-1] if result.curve else None,
    }))
    target = scene_eval.target_metrics(env, result.make_inference_fn, result.params)
    if target:
        print(json.dumps({k: round(v, 3) for k, v in target.items()}))

    if args.video:
        _render(env, result, args.video)


def _render(env: go1.Go1Env, result: train.TrainResult, out: Path, steps: int = 250) -> None:
    import jax
    import mediapy

    policy = jax.jit(result.make_inference_fn(result.params, deterministic=True))
    reset, step = jax.jit(env.reset), jax.jit(env.step)
    rng = jax.random.PRNGKey(0)
    state = reset(rng)
    traj = [state]
    for _ in range(steps):
        rng, key = jax.random.split(rng)
        action, _ = policy(state.obs, key)
        state = step(state, action)
        traj.append(state)
    camera = "overview" if env.scene.objects else "track"  # the overview shows the objects too
    frames = env.render(traj[::2], height=240, width=320, camera=camera)
    mediapy.write_video(str(out), frames, fps=1.0 / env.dt / 2)
    print(f"video: {out}")


if __name__ == "__main__":
    main()
