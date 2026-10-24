"""What a generated Go1 reward can read. No heavy imports, so fakes and prompts can use it.

The FEATURES list is the contract with the reward writer: go1.Go1Env.features()
must return exactly these keys and shapes (a test checks it).
"""

from dataclasses import dataclass

CTRL_DT = 0.02  # seconds per policy step
EPISODE_STEPS = 1000  # 20 s
STANDING_HEIGHT = 0.278  # base height in the home pose, m
LEGS = ("FR", "FL", "RR", "RL")  # front right, front left, rear right, rear left


@dataclass(frozen=True)
class Feature:
    name: str
    shape: tuple[int, ...]
    meaning: str


FEATURES: tuple[Feature, ...] = (
    Feature("base_lin_vel", (3,), "base velocity in the robot's own frame, m/s: [forward, left, up]"),
    Feature("base_ang_vel", (3,), "base angular velocity in the robot's own frame, rad/s: [roll rate, pitch rate, yaw rate]"),
    Feature("gravity", (3,), "gravity direction in the robot's frame, unit vector; [0, 0, -1] when level, x and y grow as it pitches and rolls"),
    Feature("up_z", (), "world z of the base's up axis: 1 upright, 0 on its side, below 0 upside down (the episode ends)"),
    Feature("base_height", (), f"base height above the floor, m; about {STANDING_HEIGHT} standing"),
    Feature("joint_pos", (12,), "joint angles minus the standing pose, rad; legs FR, FL, RR, RL, each (hip, thigh, calf)"),
    Feature("joint_vel", (12,), "joint velocities, rad/s, same order"),
    Feature("foot_contact", (4,), "1.0 where the foot touches the floor, else 0.0; order FR, FL, RR, RL"),
    Feature("feet_air_time", (4,), "seconds each foot has been off the floor, same order"),
    Feature("feet_height", (4,), "height of each foot above the floor, m, same order"),
    Feature("torques", (12,), "actuator torques, N·m, joint order"),
)


def describe() -> str:
    lines = [
        "Robot: Unitree Go1 quadruped in MuJoCo Playground (JAX/MJX), flat floor.",
        f"One policy step is {CTRL_DT} s; an episode is {EPISODE_STEPS} steps ({EPISODE_STEPS * CTRL_DT:.0f} s).",
        "",
        "reward(state, action, prev_action) gets:",
    ]
    for f in FEATURES:
        shape = "scalar" if not f.shape else f"shape {f.shape}"
        lines.append(f"- state[\"{f.name}\"] ({shape}): {f.meaning}")
    lines += [
        "- action, prev_action (shape (12,)): the policy's joint offsets in [-1, 1]; the motor target is "
        "the standing pose + 0.5 rad * action, joint order as above",
        "",
        "How the reward is used:",
        "- Return a dict of named scalar terms. Each step the terms are summed, multiplied by "
        f"{CTRL_DT}, and clipped below at 0. A step whose terms sum to a negative number gives 0, "
        "so keep a positive task term that outweighs the penalties (exp(-error^2 / sigma) kernels work well).",
        "- The episode ends early only if the robot flips upside down (up_z < 0).",
        "- The robot starts standing, with a random heading and small random velocities.",
        "- There is no velocity command. Reward the behavior directly.",
    ]
    return "\n".join(lines)
