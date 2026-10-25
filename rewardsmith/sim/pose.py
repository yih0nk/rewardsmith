"""Pose and gait metrics from a rollout's joint positions, for pose tasks.

"Stand on your hind legs", "lift a front paw" or "sit" are not about where the robot
goes, so the velocity metrics can't tell whether it did them. These come from the
same qpos the video is rendered from: CPU forward kinematics per step (about 20 us
each), so no change to the compiled rollout.
"""

import mujoco
import numpy as np

from .go1 import Go1Env
from .go1_spec import POSE_METRICS as METRICS  # noqa: F401  (what each metric means)

OFF_FLOOR = 0.04  # m; a foot site higher than this is off the floor (standing, it is at 6.5 mm)
ROUGH_EXTRA = 0.05  # m; rough ground has bumps up to 5 cm


def counted(done: np.ndarray) -> int:
    """Steps that count: up to and including the first fall."""
    done = np.asarray(done)
    return int(np.argmax(done)) + 1 if done.any() else len(done)


def metrics(env: Go1Env, qpos: np.ndarray) -> dict[str, float]:
    """Pose metrics for one rollout, from qpos per step (already cut to the steps that count)."""
    if hasattr(env, "pose_metrics"):  # bodies with other feet (the G1's two) measure their own
        return env.pose_metrics(qpos)
    qpos = np.asarray(qpos)
    if len(qpos) == 0:
        return {}
    model = env.mj_model
    data = mujoco.MjData(model)
    feet_ids = np.asarray(env._feet_site_id)
    torso = env._torso_body_id
    feet_z = np.empty((len(qpos), 4))
    pitch = np.empty(len(qpos))
    roll = np.empty(len(qpos))
    for i, q in enumerate(qpos):
        data.qpos[:] = q
        mujoco.mj_kinematics(model, data)
        feet_z[i] = data.site_xpos[feet_ids, 2]
        rot = data.xmat[torso].reshape(3, 3)
        pitch[i] = np.arcsin(np.clip(rot[2, 0], -1.0, 1.0))  # the forward axis' height
        roll[i] = np.arcsin(np.clip(rot[2, 1], -1.0, 1.0))  # the left axis' height
    rough = getattr(getattr(env, "scene", None), "terrain", "flat") == "rough"
    off = feet_z > OFF_FLOOR + (ROUGH_EXTRA if rough else 0.0)
    front, rear = off[:, 0] & off[:, 1], off[:, 2] & off[:, 3]  # feet order FR, FL, RR, RL
    return {
        "front_feet_off": float(front.mean()),
        "rear_feet_off": float(rear.mean()),
        "feet_off_mean": float(off.sum(axis=1).mean()),
        "front_off_held_s": _longest(front) * env.dt,
        "rear_off_held_s": _longest(rear) * env.dt,
        "pitch_deg": float(np.degrees(pitch).mean()),
        "abs_roll_deg": float(np.degrees(np.abs(roll)).mean()),
    }


def _longest(flags: np.ndarray) -> int:
    """Length of the longest run of True."""
    best = run = 0
    for f in flags:
        run = run + 1 if f else 0
        best = max(best, run)
    return best
