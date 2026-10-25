"""Roll an exported Rewardsmith policy out in MuJoCo, on the scene it trained in.

    pip install mujoco brax playground        # playground is MuJoCo Playground (imports as mujoco_playground)
    python run_policy.py                       # 500 steps (10 s) on MJX, then compare with the board's numbers
    python run_policy.py --video out.mp4       # also render the rollout (needs ffmpeg)
    python run_policy.py --viewer              # then replay it in MuJoCo's viewer (on macOS: mjpython run_policy.py --viewer)

The rollout is the board's video episode: reset key PRNGKey(--seed), one action key
split off per step, actions sampled from the policy as in training (--deterministic
takes the mean instead). On a different machine or physics backend (the board trains
with MJX's warp backend on a GPU; this defaults to the jax backend on CPU) the numbers
come out close, not bit for bit.

In steer mode the policy outputs a velocity command (forward, sideways, turn) each step
and the frozen walker in policy/walker.pkl turns it into joint targets; the env in
rewardsmith_env/go1.py does that, as it did in training.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
    os.environ.setdefault("MUJOCO_GL", "egl")  # a headless box renders --video offscreen

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from jax._src import api as _api  # noqa: E402

# brax 0.14 still calls jax.device_put_replicated, which jax 0.10 moved out of the public namespace.
for _name in ("device_put_replicated", "device_put_sharded"):
    if _name not in jax.__dict__ and hasattr(_api, _name):
        setattr(jax, _name, getattr(_api, _name))

from brax.training.acme import running_statistics  # noqa: E402
from brax.training.agents.ppo import networks as ppo_networks  # noqa: E402

from rewardsmith_env import checkpoints, go1  # noqa: E402
from rewardsmith_env.scene import parse  # noqa: E402

# The board's rollout metrics this script recomputes (render.rollout_summary).
COMPARED = ("forward_velocity", "lateral_velocity", "yaw_rate", "base_height", "upright", "fell", "rollout_steps")


def load_env(bundle: Path, impl: str):
    net = json.loads((bundle / "policy" / "network.json").read_text())
    spec = json.loads((bundle / "task" / "scene.json").read_text())
    walker = bundle / "policy" / "walker.pkl" if net["mode"] == "steer" else None
    env = go1.make_env(
        (bundle / "task" / "reward.py").read_text(), impl=impl,
        scene=parse(spec) if spec is not None else None, walker=walker,
    )
    return env, net


def load_policy(bundle: Path, env, net: dict, deterministic: bool):
    params = checkpoints.load(bundle / "policy" / "params.pkl")
    sizes = checkpoints.obs_sizes(params)
    want = {k: int(v[-1]) for k, v in env.observation_size.items()}
    if sizes != want or env.action_size != net["action_size"]:
        raise SystemExit(
            f"this env gives observations {want} and takes {env.action_size} actions; the policy was trained on "
            f"{sizes} and {net['action_size']}. Was rewardsmith_env/ replaced?"
        )
    nets = ppo_networks.make_ppo_networks(
        {k: (n,) for k, n in sizes.items()}, net["action_size"],
        preprocess_observations_fn=running_statistics.normalize,
        policy_hidden_layer_sizes=tuple(net["policy_hidden_layer_sizes"]),
        value_hidden_layer_sizes=tuple(net["value_hidden_layer_sizes"]),
        policy_obs_key=net["policy_obs_key"], value_obs_key=net["value_obs_key"],
    )
    params = jax.tree_util.tree_map(jnp.asarray, params)
    return ppo_networks.make_inference_fn(nets)(params, deterministic=deterministic)


def rollout(env, policy, steps: int, seed: int):
    """One episode as a compiled scan: the reset state, then every step's samples."""
    has_target = env.scene.target_object is not None

    def step(carry, _):
        state, rng = carry
        rng, key = jax.random.split(rng)
        action, _ = policy(state.obs, key)
        state = env.step(state, action)
        d = state.data
        out = {"qpos": d.qpos, "lin": env.get_local_linvel(d), "yaw": env.get_gyro(d)[2],
               "up": env.get_upvector(d)[2], "done": state.done, "action": action}
        if has_target:
            out["target_dist"] = env.scene_features(d)["target_dist"]
        return (state, rng), out

    @jax.jit
    def run(rng):
        state = env.reset(rng)
        d = state.data
        reset = {"qpos": d.qpos, "mocap_pos": d.mocap_pos, "mocap_quat": d.mocap_quat}
        if has_target:
            reset["target_dist"] = env.scene_features(d)["target_dist"]
        _, outs = jax.lax.scan(step, (state, rng), None, length=steps)
        return reset, outs

    reset, outs = jax.device_get(run(jax.random.PRNGKey(seed)))
    return jax.tree_util.tree_map(np.asarray, reset), jax.tree_util.tree_map(np.asarray, outs)


def summarize(reset: dict, outs: dict) -> tuple[dict, int]:
    """The board's rollout summary: only the steps up to and including the first fall count."""
    done = outs["done"].astype(bool)
    fell = bool(done.any())
    n = int(np.argmax(done)) + 1 if fell else len(done)
    out = {
        "forward_velocity": float(outs["lin"][:n, 0].mean()),
        "lateral_velocity": float(outs["lin"][:n, 1].mean()),
        "yaw_rate": float(outs["yaw"][:n].mean()),
        "base_height": float(outs["qpos"][:n, 2].mean()),
        "upright": float(outs["up"][:n].mean()),
        "fell": float(fell),
        "rollout_steps": float(n),
        "distance_m": float(np.linalg.norm(outs["qpos"][n - 1, :2] - reset["qpos"][:2])),
    }
    if "target_dist" in outs:
        out["target_dist_start"] = float(reset["target_dist"])
        out["target_dist_end"] = float(outs["target_dist"][n - 1])
        out["target_dist_min"] = float(outs["target_dist"][:n].min())
    return out, n


def render(env, reset: dict, qpos: np.ndarray, path: Path, height: int = 480, width: int = 640) -> None:
    import mediapy
    import mujoco

    model = env.mj_model
    model.vis.global_.offwidth, model.vis.global_.offheight = max(model.vis.global_.offwidth, width), max(model.vis.global_.offheight, height)
    data = mujoco.MjData(model)
    if model.nmocap:
        data.mocap_pos[:], data.mocap_quat[:] = reset["mocap_pos"], reset["mocap_quat"]
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk")
    cam.distance, cam.azimuth, cam.elevation = 1.6, 140.0, -18.0
    frames = []
    with mujoco.Renderer(model, height, width) as renderer:
        for q in qpos:
            data.qpos[:] = q
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, cam)
            frames.append(renderer.render())
    mediapy.write_video(str(path), frames, fps=round(1.0 / env.dt))


def view(env, reset: dict, qpos: np.ndarray) -> None:
    """Replay the rollout in MuJoCo's viewer, looping, at real time. Close the window to quit."""
    import mujoco
    import mujoco.viewer

    model = env.mj_model
    data = mujoco.MjData(model)
    if model.nmocap:
        data.mocap_pos[:], data.mocap_quat[:] = reset["mocap_pos"], reset["mocap_quat"]
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            for q in qpos:
                if not viewer.is_running():
                    break
                start = time.perf_counter()
                data.qpos[:] = q
                mujoco.mj_forward(model, data)
                viewer.sync()
                time.sleep(max(0.0, env.dt - (time.perf_counter() - start)))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--steps", type=int, default=500, help="control steps (500 = 10 s, the board's rollout)")
    ap.add_argument("--seed", type=int, default=0, help="reset and action key (0 = the board's video episode)")
    ap.add_argument("--deterministic", action="store_true", help="take the policy's mean action instead of sampling")
    ap.add_argument("--impl", default=None, help="MJX backend: jax or warp (default: warp on a GPU, jax elsewhere)")
    ap.add_argument("--video", type=Path, default=None, help="write the rollout to this mp4")
    ap.add_argument("--viewer", action="store_true", help="replay the rollout in MuJoCo's viewer")
    ap.add_argument("--json", type=Path, default=None, help="write the summary and the board's numbers here")
    args = ap.parse_args(argv)

    impl = args.impl or ("warp" if jax.default_backend() == "gpu" else "jax")
    env, net = load_env(HERE, impl)
    policy = load_policy(HERE, env, net, args.deterministic)
    print(f"{net['mode']} policy, {impl} backend on {jax.default_backend()}: rolling out {args.steps} steps...", flush=True)
    start = time.perf_counter()
    reset, outs = rollout(env, policy, args.steps, args.seed)
    summary, n = summarize(reset, outs)
    print(f"done in {time.perf_counter() - start:.1f} s (most of it compiling)\n")

    board = json.loads((HERE / "task" / "metrics.json").read_text())
    print(f"{'metric':<20}{'this run':>12}{'board':>12}")
    for k, v in summary.items():
        theirs = "-" if board.get(k) is None else f"{board[k]:.3f}"
        print(f"{k:<20}{v:>12.3f}{theirs:>12}")
    if args.json:
        args.json.write_text(json.dumps({"summary": summary, "board": {k: board.get(k) for k in COMPARED}}, indent=2))

    qpos = np.concatenate([reset["qpos"][None], outs["qpos"][:n]])
    if args.video:
        render(env, reset, qpos, args.video)
        print(f"\nwrote {args.video}")
    if args.viewer:
        view(env, reset, qpos)
    return 0


if __name__ == "__main__":
    sys.exit(main())
