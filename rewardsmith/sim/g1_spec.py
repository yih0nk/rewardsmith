"""What a generated Unitree G1 reward can read. No heavy imports, so fakes and prompts can use it.

The FEATURES list is the contract with the reward writer: g1.G1Env.features()
must return exactly these keys and shapes (a test checks it).
"""

from .go1_spec import Feature
from .scene import EMPTY, SceneSpec

CTRL_DT = 0.02  # seconds per policy step
EPISODE_STEPS = 1000  # 20 s
STANDING_HEIGHT = 0.755  # pelvis height in the knees-bent home pose, m
TORSO_HEIGHT = 0.80  # torso link origin standing, m; the head tops out near 1.2 m
NU = 29  # actuated joints
JOINTS = (
    "left leg (hip pitch, hip roll, hip yaw, knee, ankle pitch, ankle roll), right leg (same 6), "
    "waist (yaw, roll, pitch), left arm (shoulder pitch, shoulder roll, shoulder yaw, elbow, wrist roll, "
    "wrist pitch, wrist yaw), right arm (same 7)"
)

# The episode ends when the robot has fallen: torso tilted past ~70 degrees or pelvis this low.
FALL_UP_Z = 0.35
FALL_HEIGHT = 0.35  # m

# Steer mode: the policy's action in [-1, 1]^3 times these is the walker's command target
# (forward m/s, left m/s, yaw rad/s): Playground's G1 command ranges, what the walker learned.
COMMAND_LIMITS = (1.0, 0.5, 1.0)
COMMAND_SMOOTHING = 0.2

# Playground's tuned G1 joystick terms (g1/joystick.default_config) that keep the gait and posture
# clean, with their stock scales. Left out: the command-tracking terms and stand_still (the command
# is always 0 outside steer mode), and termination (-100, which the clip at 0 would make meaningless).
STOCK_SCALES: dict[str, float] = {
    "ang_vel_xy": -0.15, "orientation": -2.0, "feet_air_time": 2.0, "feet_slip": -0.25, "feet_phase": 1.0,
    "collision": -0.1, "contact_force": -0.01, "joint_deviation_knee": -0.1, "joint_deviation_hip": -0.25,
    "dof_pos_limits": -1.0, "pose": -0.1,
}
POSE_BREAKS = ("ang_vel_xy", "orientation", "feet_air_time", "feet_slip", "feet_phase", "joint_deviation_knee",
               "joint_deviation_hip", "pose")
STOCK_PROFILES: dict[str, dict[str, float]] = {
    "gait": {},
    "pose": {k: 0.0 for k in POSE_BREAKS},
    "off": {k: 0.0 for k in STOCK_SCALES},
}
STOCK_WEIGHT = 1.0
STOCK_PREFIX = "stock_"
STOCK_MEANINGS: dict[str, str] = {
    "ang_vel_xy": "torso roll and pitch rates squared",
    "orientation": "torso tilt away from upright (slightly forward)",
    "feet_air_time": "air time minus 0.2 s at each touchdown, capped (rewards real steps)",
    "feet_slip": "body speed while a foot is in contact",
    "feet_phase": "feet following the walker's alternating step clock, 15 cm swing height",
    "collision": "a hand touching its thigh",
    "contact_force": "foot impact force above 500 N",
    "joint_deviation_knee": "knees away from the home pose",
    "joint_deviation_hip": "hip roll and yaw away from the home pose",
    "dof_pos_limits": "joint angles past 95% of their range",
    "pose": "squared distance of all joints from the home pose",
}

FEATURES: tuple[Feature, ...] = (
    Feature("base_lin_vel", (3,), "pelvis velocity in the robot's own frame, m/s: [forward, left, up]"),
    Feature("base_ang_vel", (3,), "pelvis angular velocity in its own frame, rad/s: [roll rate, pitch rate, yaw rate]"),
    Feature("gravity", (3,), "gravity direction in the pelvis frame, unit vector; [0, 0, -1] when level"),
    Feature("up_z", (), f"world z of the torso's up axis: 1 upright, 0 lying down; the episode ends below {FALL_UP_Z}"),
    Feature("base_height", (), f"pelvis height above the floor, m; about {STANDING_HEIGHT} standing; the episode ends below {FALL_HEIGHT}"),
    Feature("torso_height", (), f"height of the torso (chest) above the floor, m; about {TORSO_HEIGHT} standing"),
    Feature("joint_pos", (NU,), f"joint angles minus the home pose, rad; order: {JOINTS}"),
    Feature("joint_vel", (NU,), "joint velocities, rad/s, same order"),
    Feature("foot_contact", (2,), "1.0 where the foot touches the floor, else 0.0; order left, right"),
    Feature("feet_air_time", (2,), "seconds each foot has been off the floor, order left, right"),
    Feature("feet_height", (2,), "height of each foot above the floor, m (about 0.03 standing), order left, right"),
    Feature("hand_height", (2,), "height of each palm above the floor, m (about 0.66 with arms down), order left, right"),
    Feature("torques", (NU,), "actuator torques, N·m, joint order"),
    Feature("base_pos", (3,), "pelvis position in the world, m: [x, y, z]; each episode starts within 0.5 m of [0, 0]"),
    Feature("base_yaw", (), "the robot's heading in the world, rad, in [-pi, pi]; random at the start of each episode"),
    Feature("time", (), "seconds since the episode started"),
)

# Rollout gait/pose metrics (G1Env.pose_metrics), over the steps up to the first fall:
# name -> (meaning, lowest sensible, highest). pitch_deg/abs_roll_deg mean what they do for the Go1.
POSE_METRICS: dict[str, tuple[str, float, float]] = {
    "single_support": ("humanoid: fraction of the rollout with exactly one foot on the floor (walking is about 0.6-0.8)",
                       0.0, 1.0),
    "double_support": ("humanoid: fraction of the rollout with both feet on the floor (standing still is about 1)",
                       0.0, 1.0),
    "flight": ("humanoid: fraction of the rollout with BOTH feet off the floor (hopping, jumping, running)", 0.0, 1.0),
    "flight_held_s": ("humanoid: longest unbroken stretch with both feet off the floor, s (a jump is 0.1-0.4)",
                      0.0, 20.0),
    "step_count": ("humanoid: number of foot touchdowns in the rollout (walking: about 2 per second)", 0.0, 1000.0),
    "pitch_deg": ("mean torso pitch, degrees; positive = leaning back, negative = leaning forward", -90.0, 90.0),
    "abs_roll_deg": ("mean absolute torso roll, degrees (0 upright, 90 on its side)", 0.0, 180.0),
}


def features_for(scene: SceneSpec = EMPTY) -> tuple[Feature, ...]:
    return FEATURES


def stock_lines(profile: str, weight: float = STOCK_WEIGHT) -> list[str]:
    defaults = STOCK_PROFILES[profile]
    on = [k for k in STOCK_SCALES if defaults.get(k, 1.0) != 0.0]
    if not on:
        return []
    lines = [
        "",
        "Stock gait terms: under your terms the env adds Playground's tuned G1 walking terms (the ones the "
        "pretrained walker learned with), so the robot keeps a steady two-legged gait while it learns your task. "
        "They show up as stock_<name> in the training logs. On for this request, with their per-step scales:",
    ]
    lines += [f"- {k} ({weight * STOCK_SCALES[k]:+g}): {STOCK_MEANINGS[k]}" for k in on]
    off = [k for k in STOCK_SCALES if k not in on]
    if off:
        lines.append(f"Off for this request (the behavior tilts the body or leaves the gait): {', '.join(off)}.")
    lines.append(
        "To rescale or drop stock terms, define a module-level dict next to reward(), multipliers on the "
        "scales above (0 drops a term), e.g. STOCK_TERMS = {\"orientation\": 0.0, \"pose\": 0.0}. "
        "Don't name your own terms stock_*."
    )
    return lines


def describe(
    scene: SceneSpec = EMPTY, *, mode: str = "off", stock_profile: str = "off", stock_weight: float = STOCK_WEIGHT
) -> str:
    """The writer's notes. mode is how the policy uses the pretrained walker (playground.MODES)."""
    floor = "rough, bumpy ground" if scene.terrain == "rough" else "flat floor"
    lines = [
        f"Robot: Unitree G1 humanoid (about 1.3 m tall, 29 joints: legs, waist, arms; no hand control) in MuJoCo "
        f"Playground (JAX/MJX), {floor}. It walks on two legs.",
        f"One policy step is {CTRL_DT} s; an episode is {EPISODE_STEPS} steps ({EPISODE_STEPS * CTRL_DT:.0f} s).",
        "",
        "reward(state, action, prev_action) gets:",
    ]
    for f in FEATURES:
        shape = "scalar" if not f.shape else f"shape {f.shape}"
        lines.append(f"- state[\"{f.name}\"] ({shape}): {f.meaning}")
    lines += [
        f"- action, prev_action (shape ({NU},)): joint offsets in [-1, 1]; the motor target is the home pose "
        "+ 0.5 rad * action, joint order as above",
        "- Torso pitch is arcsin(-gravity[0]) and roll arcsin(-gravity[1]) (pelvis frame), in radians; a foot is "
        "off the floor when its foot_contact is 0.",
        "",
        "How the reward is used:",
        f"- Return a dict of named scalar terms. Each step the terms are summed, multiplied by {CTRL_DT}, and "
        "clipped below at 0, so keep a positive task term that outweighs the penalties "
        "(exp(-error^2 / sigma) kernels work well).",
        f"- The robot has fallen, and the episode ends, when up_z < {FALL_UP_Z} or base_height < {FALL_HEIGHT} m, "
        "or a foot hits the other leg.",
        "- The robot starts standing with knees bent, a random heading and small random velocities.",
    ]
    if mode != "steer":
        lines.append("- There is no velocity command. Reward the behavior directly. Balancing on two legs is "
                     "hard: keep an upright/alive term (e.g. up_z and base_height near standing).")
    if mode == "tune":
        lines.append(
            "- The policy starts from a pretrained G1 walker (Playground's joystick policy): it already balances "
            "and steps in place, with its command at 0. Your reward teaches the task, not balance."
        )
    elif mode == "steer":
        fwd, side, turn = COMMAND_LIMITS
        lines.append(
            "- The policy steers a pretrained walker (Playground's G1 joystick policy, a stable two-legged gait): "
            f"each step it picks a velocity command, forward/backward up to {fwd:g} m/s, sideways up to {side:g} m/s "
            f"and turning up to {turn:g} rad/s, which eases toward the pick ({COMMAND_SMOOTHING:.0%} per step), and "
            "the walker moves the legs (at command 0 it steps in place). So reward the outcome (where the robot "
            "goes and faces, how fast); the policy cannot change how the legs move, so gait, posture, foot and "
            "smoothness terms have no effect: leave them out. action and prev_action are the walker's joint actions."
        )
    lines += stock_lines(stock_profile, stock_weight)
    return "\n".join(lines)
