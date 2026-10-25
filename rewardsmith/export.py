"""Export one trained candidate as a zip that runs anywhere MuJoCo Playground does.

    <name>/README.md               what is in it and how to run it
    <name>/run_policy.py           rolls the policy out in MuJoCo on the candidate's scene
    <name>/manifest.json           which candidate, its mode and checkpoint, what it scored
    <name>/policy/params.pkl       brax PPO params (normalizer, policy, value) as numpy arrays
    <name>/policy/walker.pkl       steer mode only: the frozen walker the policy commands
    <name>/policy/network.json     the brax network config that rebuilds the inference function
    <name>/task/reward.py          the reward it trained on
    <name>/task/scene.json         the scene spec (null: the plain flat floor)
    <name>/task/success_check.json the request's pass/fail check and this run's result
    <name>/task/metrics.json       the board's metrics for this run
    <name>/task/critique.json      the critic's score and notes
    <name>/rollout.mp4             the board's video
    <name>/rewardsmith_env/        a copy of the Go1 env code (go1.py and what it imports)

The env code is copied rather than rewritten so the rollout sees exactly the
observations the policy trained on: scene features, terrain inputs and steer
mode's command smoothing all live in go1.py.

Kept free of jax and brax imports: the API server builds these.
"""

import json
import os
import time
import zipfile
from pathlib import Path
from typing import IO, Any

from .sim import checkpoints, progress

# The walker steer-mode candidates command. Matches sim/playground.py's default (which imports jax).
DEFAULT_WALKER = "/workspace/pretrained/go1_walker/params.pkl"

# Playground's Go1 joystick network (locomotion_params.brax_ppo_config(go1.BASE_ENV).network_factory),
# which train.ppo_kwargs trains every candidate with. tests/test_export.py checks they still match.
NETWORK = {
    "policy_hidden_layer_sizes": [512, 256, 128],
    "value_hidden_layer_sizes": [512, 256, 128],
    "policy_obs_key": "state",
    "value_obs_key": "privileged_state",
}
JOINT_ACTIONS = 12
STEER_ACTIONS = 3  # (vx, vy, yaw rate): go1.COMMAND_LIMITS

# go1.py and everything it imports from the package, as run_policy.py imports them.
_SIM_DIR = Path(__file__).resolve().parent / "sim"
ENV_FILES = ("go1.py", "go1_spec.py", "scene.py", "scene_xml.py", "reward_slot.py", "warp_quiet.py", "checkpoints.py")
_TEMPLATES = Path(__file__).resolve().parent / "export_files"


class NotExportable(Exception):
    """The candidate has no policy on disk (not trained, or its files are gone)."""


def walker_path() -> Path:
    return Path(os.environ.get("REWARDSMITH_WALKER", DEFAULT_WALKER))


def policy_file(out_dir: Path) -> Path | None:
    """The candidate's final params: params.pkl if training wrote one, else its latest params_<steps>.pkl."""
    final = Path(out_dir) / "params.pkl"
    return final if final.is_file() else checkpoints.latest(out_dir)


def bundle_name(cand: dict[str, Any]) -> str:
    return f"rewardsmith-{cand['request_id']}-t{cand.get('turn') or 1}r{cand['round_n']}c{cand['idx']}"


def write_zip(
    fileobj: IO[bytes], cand: dict[str, Any], req: dict[str, Any], artifacts_dir: Path, walker: Path | None = None,
) -> str:
    """Write the candidate's bundle to fileobj; return the bundle's name.

    cand is db.get_candidate's row, req its request (db.get_request). Raises
    NotExportable when there is no policy to export.
    """
    out_dir = progress.candidate_dir(artifacts_dir, cand["request_id"], cand["round_n"], cand["idx"])
    params = policy_file(out_dir)
    if params is None:
        raise NotExportable(f"Candidate {cand['id']} has no trained policy yet")
    metrics = cand.get("metrics") or {}
    steer = bool(metrics.get("steered"))
    walker = walker or walker_path()
    if steer and not walker.is_file():
        raise NotExportable(f"Candidate {cand['id']} steers the walker, and there is no walker at {walker}")

    name = bundle_name(cand)
    reward = (out_dir / "reward.py").read_text() if (out_dir / "reward.py").is_file() else cand["reward_src"]
    video = artifacts_dir / cand["video_path"] if cand.get("video_path") else None
    manifest = {
        "candidate_id": cand["id"],
        "request_id": cand["request_id"],
        "turn": cand.get("turn") or 1,
        "round": cand["round_n"],
        "candidate": cand["idx"],
        "task": req.get("task") or req.get("text"),
        "score": cand.get("score"),
        "mode": "steer" if steer else "joints",
        "action_size": STEER_ACTIONS if steer else JOINT_ACTIONS,
        "checkpoint": params.name,
        "train_steps": metrics.get("train_steps"),
        "exported_at": round(time.time(), 3),
    }
    with zipfile.ZipFile(fileobj, "w", zipfile.ZIP_DEFLATED) as z:
        def text(path: str, body: str) -> None:
            z.writestr(f"{name}/{path}", body)

        def as_json(path: str, data: Any) -> None:
            text(path, json.dumps(data, indent=2) + "\n")

        text("README.md", (_TEMPLATES / "README.md").read_text().replace("{{TASK}}", manifest["task"] or ""))
        z.write(_TEMPLATES / "run_policy.py", f"{name}/run_policy.py")
        as_json("manifest.json", manifest)
        z.write(params, f"{name}/policy/params.pkl", zipfile.ZIP_STORED)  # numpy floats barely compress
        if steer:
            z.write(walker, f"{name}/policy/walker.pkl", zipfile.ZIP_STORED)
        as_json("policy/network.json", {**NETWORK, "action_size": manifest["action_size"], "mode": manifest["mode"]})
        text("task/reward.py", reward)
        as_json("task/scene.json", req.get("scene"))
        as_json("task/success_check.json", {
            "spec": req.get("success_spec"), "result": cand.get("check_result"),
            "error": req.get("success_check_error"),
        })
        as_json("task/metrics.json", metrics)
        as_json("task/critique.json", {
            "score": cand.get("score"), "critique": cand.get("critique"), "idea": cand.get("idea"),
            "reward_hacking": cand.get("reward_hacking"),
        })
        if video is not None and video.is_file():
            z.write(video, f"{name}/rollout.mp4", zipfile.ZIP_STORED)
        text("rewardsmith_env/__init__.py", '"""A copy of rewardsmith/sim\'s Go1 env, for run_policy.py."""\n')
        for f in ENV_FILES:
            z.write(_SIM_DIR / f, f"{name}/rewardsmith_env/{f}")
    return name
