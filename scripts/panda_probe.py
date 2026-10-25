"""Train the Panda on a reward file and print its rollout metrics: how big a budget the arm needs.

    python scripts/panda_probe.py <reward.py> [--steps 10000000] [--table] [--joint] [--ppo '{"num_envs": 2048}']
"""

import argparse
import functools
import json
import time
from pathlib import Path

from rewardsmith.sim import jax_compat  # noqa: F401  (must come before brax's PPO trainer)
import jax
import numpy as np
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from mujoco_playground import wrapper
from mujoco_playground.config import manipulation_params

from rewardsmith.sim import panda
from rewardsmith.sim.scene import parse

TABLE_SCENE = json.dumps({
    "objects": [
        {"name": "cube", "kind": "box", "color": "red", "size": [0.04, 0.04, 0.06], "distance": 1.0, "bearing": 0,
         "movable": True, "mass": 0.1},
        {"name": "goal", "kind": "zone", "color": "blue", "size": [0.3], "distance": 2.0, "bearing": 0},
    ],
    "target": "cube", "goal": "goal",
})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("reward", type=Path)
    ap.add_argument("--steps", type=int, default=10_000_000)
    ap.add_argument("--table", action="store_true")
    ap.add_argument("--joint", action="store_true")
    ap.add_argument("--ppo", type=json.loads, default={})
    args = ap.parse_args()
    scene = parse(TABLE_SCENE) if args.table else None
    env = panda.make_env(args.reward.read_text(), scene=scene, ee=not args.joint, behavior_metrics=True)
    cfg = manipulation_params.brax_ppo_config(panda.BASE_ENV)
    kw = {k: v for k, v in cfg.items() if k != "network_factory"}
    kw.update(num_timesteps=args.steps, episode_length=panda.TRAIN_EPISODE_STEPS, num_evals=6, num_resets_per_eval=1,
              num_eval_envs=128)
    kw.update(args.ppo)
    net = dict(cfg.network_factory) | kw.pop("network", {})
    t0 = time.perf_counter()

    def progress(steps, m):
        L = float(m.get("eval/avg_episode_length", 1))
        beh = {k.split("/")[-1]: round(float(v) / L, 3) for k, v in m.items() if "episode_behavior/" in k and not k.endswith("_std")}
        print(f"{time.perf_counter() - t0:7.1f}s {steps:>10} reward={float(m['eval/episode_reward']):8.2f} len={L:.0f} {beh}", flush=True)

    make_inf, params, _ = ppo.train(
        environment=env, wrap_env_fn=wrapper.wrap_for_brax_training, progress_fn=progress, seed=0,
        network_factory=functools.partial(ppo_networks.make_ppo_networks, **net), **kw,
    )
    print(f"train {time.perf_counter() - t0:.1f}s", flush=True)

    eval_env = panda.make_env(args.reward.read_text(), scene=scene, ee=not args.joint)
    policy = make_inf(params, deterministic=True)
    n = 32

    def one(rng):
        state = eval_env.reset(rng)

        def body(s, _):
            a, _ = policy(s.obs, rng)
            s = eval_env.step(s, a)
            return s, eval_env.frame(s.data, s.info)

        _, frames = jax.lax.scan(body, state, None, length=panda.EPISODE_STEPS)
        return frames

    frames = jax.jit(jax.vmap(one))(jax.random.split(jax.random.PRNGKey(7), n))
    frames = jax.tree_util.tree_map(np.asarray, frames)
    print("metrics", json.dumps({k: round(v, 3) for k, v in panda.rollout_metrics(frames).items()}), flush=True)


if __name__ == "__main__":
    main()
