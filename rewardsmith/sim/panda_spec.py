"""What a generated Franka Panda reward can read. No heavy imports, so fakes and prompts can use it.

The FEATURES list is the contract with the reward writer: panda.PandaEnv.features()
returns exactly these keys and shapes (a test checks it).

The arm stands at the world origin on a flat table (the floor plane, z = 0), facing +x.
One cube rests on the table in front of it; a target point (shown as a faint marker)
says where the cube should go: in the air for lift tasks, on the table for push and
place tasks.
"""

from dataclasses import dataclass

from .scene import EMPTY, SceneSpec

CTRL_DT = 0.02  # seconds per policy step
EPISODE_STEPS = 500  # 10 s, the rollout and evaluation length
TRAIN_EPISODE_STEPS = EPISODE_STEPS  # train on the whole 10 s: at 6 s policies let go after 6 s
CUBE_SIZE = (0.02, 0.02, 0.03)  # m, half sizes: a 4 x 4 x 6 cm block (Playground's pick-cube block)
CUBE_REST_Z = CUBE_SIZE[2]  # block centre height resting on the table
REACH_M = 0.85
# Where the cube starts: x, y around (0.5, 0) +- this, m.
CUBE_X, CUBE_Y, CUBE_RANGE = 0.5, 0.0, 0.15
# The end-effector skill: action in [-1, 1]^4 -> (dx, dy, dz) * EE_STEP per step, and the gripper.
EE_STEP = 0.003  # m per step = 0.15 m/s at full action; faster and the arm lags and swings
# The gripper target is kept in this box (m, world frame), inside the arm's analytic-IK workspace.
EE_BOUNDS = ((0.25, 0.77), (-0.35, 0.35), (0.0, 0.5))
FINGER_OPEN = 0.04  # m per finger; the opening is the sum of both, 0 to 0.08
LIFT_HEIGHT = 0.05  # m above the table that counts as lifted (cube bottom)
GRIPPER_DOWN_OFFSET = 0.0  # the gripper site sits between the fingertips' pads


@dataclass(frozen=True)
class Feature:
    name: str
    shape: tuple[int, ...]
    meaning: str


FEATURES: tuple[Feature, ...] = (
    Feature("gripper_pos", (3,), "gripper (point between the finger pads) in the world, m: [x forward, y left, z up]; "
                                 "starts about [0.68, 0, 0.38]"),
    Feature("gripper_vel", (3,), "gripper velocity in the world, m/s"),
    Feature("gripper_opening", (), "distance between the fingers, m: 0.08 fully open, 0 closed; about 0.04 when "
                                   "closed on the cube"),
    Feature("finger_contact", (2,), "1.0 where the [left, right] finger pad touches the cube, else 0.0"),
    Feature("grasped", (), "1.0 when both finger pads touch the cube (it is held), else 0.0"),
    Feature("hand_floor_contact", (), "1.0 when the fingers or hand touch the table, else 0.0 (avoid it)"),
    Feature("cube_pos", (3,), "cube centre in the world, m; resting on the table its z is 0.03"),
    Feature("cube_vel", (3,), "cube velocity in the world, m/s"),
    Feature("cube_height", (), "block bottom above the table, m: 0 resting, > 0.05 lifted"),
    Feature("cube_up_z", (), "world z of the cube's up axis: 1 upright, 0 on its side"),
    Feature("cube_to_gripper", (3,), "gripper minus cube centre, m, world frame"),
    Feature("gripper_to_cube_dist", (), "distance from the gripper to the cube centre, m; under 0.02 = in the "
                                        "fingers"),
    Feature("target_pos", (3,), "target point in the world, m (where the cube should end up)"),
    Feature("cube_to_target", (3,), "target minus cube centre, m, world frame: the direction the cube must go"),
    Feature("cube_to_target_dist", (), "distance from the cube centre to the target, m; the number to drive to 0"),
    Feature("joint_pos", (7,), "arm joint angles, rad, shoulder to wrist"),
    Feature("joint_vel", (7,), "arm joint velocities, rad/s"),
    Feature("time", (), "seconds since the episode started"),
)

# Rollout metrics for the success check (sim/panda.py rollout_metrics): name -> (meaning, low, high).
# Rates are over the evaluation episodes; the rest are means over them.
ARM_METRICS: dict[str, tuple[str, float, float]] = {
    "lifted": ("fraction of episodes where the cube bottom rose at least 5 cm above the table while held", 0.0, 1.0),
    "lift_held_s": ("mean longest unbroken stretch with the cube lifted and held, s", 0.0, 10.0),
    "max_cube_height": ("mean highest cube-bottom height above the table, m", 0.0, 0.6),
    "grasped_frac": ("mean fraction of the episode with the cube held in both fingers", 0.0, 1.0),
    "final_target_dist": ("mean cube-to-target distance at episode end, m", 0.0, 1.5),
    "min_target_dist": ("mean closest the cube got to the target, m", 0.0, 1.5),
    "at_target": ("fraction of episodes ending with the cube within 5 cm of the target", 0.0, 1.0),
    "placed": ("fraction of episodes ending with the cube resting on the table (not held) within 5 cm of a "
               "table-level target", 0.0, 1.0),
    "dropped": ("fraction of episodes where the cube was lifted, then fell back to the table away from the target", 0.0, 1.0),
    "cube_moved": ("mean horizontal distance the cube ended from where it started, m", 0.0, 1.5),
    "fell": ("fraction of episodes that ended early: cube knocked off the table or a joint limit hit", 0.0, 1.0),
}
AT_TARGET_M = 0.05


def target_mode(scene: SceneSpec = EMPTY) -> str:
    """"table" (push/place: the target sits on the table) or "air" (lift: above it)."""
    return "table" if scene.goal_object is not None else "air"


def scene_notes(scene: SceneSpec = EMPTY) -> list[str]:
    """What the arm makes of a scene spec: only the target cube's color and a goal zone are used."""
    lines = []
    target = scene.target_object
    color = target.color if target is not None else "green"
    lines.append(f"- One {color} block (4 x 4 cm, 6 cm tall, \"the cube\") starts on the table at x = {CUBE_X:g} +- {CUBE_RANGE:g} m, "
                 f"y = {CUBE_Y:g} +- {CUBE_RANGE:g} m, upright, a random spot each episode.")
    if target_mode(scene) == "table":
        lines.append("- The target is a spot on the table (cube centre at z = 0.03), 10 to 25 cm from the cube's "
                     "start in a random direction each episode: slide or carry the cube there and let it rest.")
    else:
        lines.append("- The target is a point 15 to 30 cm above the table, near the cube's start: pick the cube up "
                     "and hold it there.")
    return lines


def describe(scene: SceneSpec = EMPTY, *, mode: str = "steer") -> str:
    """The writer's notes. mode "steer" drives the end-effector skill; anything else, the joints."""
    lines = [
        "Robot: Franka Emika Panda 7-joint arm with a parallel two-finger gripper, MuJoCo Playground (JAX/MJX), "
        "bolted to a flat table at the world origin, facing +x. It has no legs and cannot move its base.",
        f"One policy step is {CTRL_DT} s; an evaluation episode is {EPISODE_STEPS} steps "
        f"({EPISODE_STEPS * CTRL_DT:.0f} s), the same in training.",
        "",
        "Scene (fixed for the arm; walls, ramps, balls and other scene objects are not supported):",
        *scene_notes(scene),
        "",
        "reward(state, action, prev_action) gets:",
    ]
    for f in FEATURES:
        shape = "scalar" if not f.shape else f"shape {f.shape}"
        lines.append(f"- state[\"{f.name}\"] ({shape}): {f.meaning}")
    if mode == "steer":
        lines.append(
            f"- action, prev_action (shape (4,)): the end-effector controller's command in [-1, 1]: [dx, dy, dz] move "
            f"the gripper target up to {EE_STEP * 1000:g} mm per step ({EE_STEP / CTRL_DT:g} m/s) in the world frame, "
            "the gripper pointing straight down; action[3] < 0 closes the fingers, >= 0 opens them. Inverse "
            "kinematics turns it into joint targets, so reward where the gripper and cube go, not joint motion."
        )
    else:
        lines.append("- action, prev_action (shape (8,)): joint target offsets in [-1, 1] (7 arm joints, then the "
                     "gripper), 0.04 rad per unit per step.")
    lines += [
        "",
        "How the reward is used:",
        "- Return a dict of named scalar terms. Each step the terms are summed (no dt factor) and that is the "
        "step's reward. Keep each term in about [-1, 1] per step; bounded kernels like 1 - tanh(5 * dist) work "
        "well. A good pick reward stages: reach the cube, close the fingers on it, then lift or carry it, e.g. "
        "gating the carry term on state[\"grasped\"].",
        "- The episode ends early (\"fell\") if the cube leaves the table (more than 1 m from the arm, or below it) "
        "or the physics blows up.",
        "- The arm starts in its home pose above the table, fingers open, at rest.",
        "- The policy observes the gripper, the cube, the target and the finger contacts, so reward the skill "
        "relative to them, not a fixed spot.",
    ]
    return "\n".join(lines)
