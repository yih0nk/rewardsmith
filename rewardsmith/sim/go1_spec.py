"""What a generated Go1 reward can read. No heavy imports, so fakes and prompts can use it.

The FEATURES list is the contract with the reward writer: go1.Go1Env.features()
must return exactly these keys and shapes (a test checks it). A scene with
objects adds per-object features on top: features_for(scene) lists them all.
"""

from dataclasses import dataclass

from .scene import EMPTY, SceneObject, SceneSpec

CTRL_DT = 0.02  # seconds per policy step
EPISODE_STEPS = 1000  # 20 s
STANDING_HEIGHT = 0.278  # base height in the home pose, m
LEGS = ("FR", "FL", "RR", "RL")  # front right, front left, rear right, rear left


# Playground's tuned Go1 joystick reward terms (joystick.default_config) that keep the gait clean and
# the body steady, with their stock scales. Left out: tracking_lin_vel, tracking_ang_vel and
# stand_still, which reward following the joystick command, and that is always 0 here.
STOCK_SCALES: dict[str, float] = {
    "lin_vel_z": -0.5, "ang_vel_xy": -0.05, "orientation": -5.0, "dof_pos_limits": -1.0, "pose": 0.5,
    "termination": -1.0, "torques": -0.0002, "action_rate": -0.01, "energy": -0.001,
    "feet_clearance": -2.0, "feet_height": -0.2, "feet_slip": -0.1, "feet_air_time": 0.1,
}
# The terms a pose or a jump has to break: the body tilts, bobs, leaves the standing pose and lifts its feet.
POSE_BREAKS = ("lin_vel_z", "ang_vel_xy", "orientation", "pose", "feet_clearance", "feet_height", "feet_slip",
               "feet_air_time")
# Per-term multipliers by request family (sim/budget.py picks one); a reward's STOCK_TERMS overrides them.
STOCK_PROFILES: dict[str, dict[str, float]] = {
    "gait": {},
    "pose": {k: 0.0 for k in POSE_BREAKS},
    "off": {k: 0.0 for k in STOCK_SCALES},
}
STOCK_WEIGHT = 1.0  # every stock term's scale is multiplied by this
STOCK_PREFIX = "stock_"

# What each stock term measures, for the writer.
STOCK_MEANINGS: dict[str, str] = {
    "lin_vel_z": "base vertical velocity squared",
    "ang_vel_xy": "base roll and pitch rates squared",
    "orientation": "body tilt: squared x and y of the base's up axis",
    "dof_pos_limits": "joint angles past 95% of their limits",
    "pose": "exp(-distance from the standing pose), hips and thighs weighted 10x the calves",
    "termination": "1 on the step the robot flips",
    "torques": "actuator torque magnitudes",
    "action_rate": "squared change of the action between steps",
    "energy": "|joint velocity| x |torque|, summed",
    "feet_clearance": "swing feet away from 10 cm high, weighted by foot speed",
    "feet_height": "swing peak away from 10 cm, at each touchdown",
    "feet_slip": "foot speed while in contact",
    "feet_air_time": "air time minus 0.1 s at each touchdown (rewards real steps, punishes shuffling)",
}


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
    Feature("base_pos", (3,), "base position in the world, m: [x, y, z]; each episode starts within 0.5 m of [0, 0]"),
    Feature("base_yaw", (), "the robot's heading in the world, rad, in [-pi, pi]; random at the start of each episode"),
    Feature("time", (), "seconds since the episode started"),
)

# Rollout pose metrics (sim/pose.py), for pose tasks: name -> (meaning, lowest sensible, highest).
# All are over the rollout's steps up to the first fall.
POSE_METRICS: dict[str, tuple[str, float, float]] = {
    "front_feet_off": ("fraction of the rollout with BOTH front feet off the floor (e.g. standing on the hind legs)",
                       0.0, 1.0),
    "rear_feet_off": ("fraction of the rollout with BOTH rear feet off the floor (e.g. a handstand on the front legs)",
                      0.0, 1.0),
    "feet_off_mean": ("mean number of feet off the floor, 0 to 4 (about 2 in a trot, 0 standing)", 0.0, 4.0),
    "front_off_held_s": ("longest unbroken stretch with both front feet off the floor, s", 0.0, 20.0),
    "rear_off_held_s": ("longest unbroken stretch with both rear feet off the floor, s", 0.0, 20.0),
    "pitch_deg": ("mean body pitch, degrees; positive = nose up (about 60 to 90 standing on the hind legs)",
                  -90.0, 90.0),
    "abs_roll_deg": ("mean absolute body roll, degrees (0 level, 90 on its side)", 0.0, 180.0),
}

# Per-object features: state[f"{name}_{suffix}"], and state[f"target_{suffix}"] for the target object.
OBJECT_FEATURES: tuple[tuple[str, tuple[int, ...], str], ...] = (
    ("rel_pos", (3,), "object centre relative to the robot base, m, in the robot's heading frame: "
                      "[forward, left, up]; turning with the robot, ignoring its pitch and roll"),
    ("dist", (), "horizontal distance from the robot base to the object centre, m"),
    ("heading_err", (), "angle from the robot's forward direction to the object, rad, in [-pi, pi]; "
                        "0 = straight ahead, positive = to the left"),
    ("pos", (3,), "object centre in the world, m: [x, y, z]"),
)
MOVABLE_FEATURES: tuple[tuple[str, tuple[int, ...], str], ...] = (
    ("vel", (3,), "object velocity in the world, m/s: [x, y, z]"),
)
# With a goal (move the target to it): where the goal is, seen from the target object.
GOAL_FEATURES: tuple[Feature, ...] = (
    Feature("target_to_goal_rel", (3,), "goal centre minus target centre, m, in the robot's heading frame: "
                                        "[forward, left, up]; the direction the target must travel"),
    Feature("target_to_goal_dist", (), "horizontal distance from the target object's centre to the goal's centre, m; "
                                       "the number to drive to zero"),
    Feature("target_to_goal_speed", (), "how fast the target object closes on the goal, m/s (minus the rate of change of "
                                        "target_to_goal_dist): positive while it moves toward the goal, negative away. "
                                        "Rewarding it is the potential-based progress term"),
)


def object_features(obj: SceneObject, alias: str | None = None) -> tuple[Feature, ...]:
    """Features for one object, named after it (or after alias, e.g. "target")."""
    rows = OBJECT_FEATURES + (MOVABLE_FEATURES if obj.movable else ())
    who = alias or obj.name
    return tuple(Feature(f"{who}_{suffix}", shape, meaning) for suffix, shape, meaning in rows)


def scene_features(scene: SceneSpec) -> tuple[Feature, ...]:
    out: list[Feature] = []
    for obj in scene.objects:
        out += object_features(obj)
    if scene.target_object is not None:
        out += object_features(scene.target_object, alias="target")
    if scene.goal_object is not None:
        out += GOAL_FEATURES
    return tuple(out)


def features_for(scene: SceneSpec = EMPTY) -> tuple[Feature, ...]:
    return FEATURES + scene_features(scene)


def stock_lines(profile: str, weight: float = STOCK_WEIGHT) -> list[str]:
    """The writer's notes on the stock terms under a profile (empty when they are all off)."""
    defaults = STOCK_PROFILES[profile]
    on = [k for k in STOCK_SCALES if defaults.get(k, 1.0) != 0.0]
    if not on:
        return []
    lines = [
        "",
        "Stock gait terms: under your terms the env adds Playground's tuned Go1 walking terms (the ones the "
        "pretrained walker learned with), so the robot keeps a clean, steady gait while it learns your task. "
        "They show up as stock_<name> in the training logs. On for this request, with their per-step scales:",
    ]
    lines += [f"- {k} ({weight * STOCK_SCALES[k]:+g}): {STOCK_MEANINGS[k]}" for k in on]
    off = [k for k in STOCK_SCALES if k not in on]
    if off:
        lines.append(f"Off for this request (the behavior tilts or lifts the body): {', '.join(off)}.")
    lines += [
        "To rescale or drop stock terms, define a module-level dict next to reward(), multipliers on the "
        "scales above (0 drops a term, 2 doubles it), e.g. STOCK_TERMS = {\"orientation\": 0.0, \"pose\": 0.0}. "
        "Drop orientation, pose, lin_vel_z and ang_vel_xy for behaviors that tilt or bob the body (standing on "
        "hind legs, sitting, jumping, rolling over); keep them for walking, turning, reaching and pushing. "
        "Don't name your own terms stock_*. Don't re-add smoothness or posture penalties the stock terms "
        "already cover; spend your terms on the task.",
    ]
    return lines


def describe(
    scene: SceneSpec = EMPTY, *, warm_start: bool = False, stock_profile: str = "off", stock_weight: float = STOCK_WEIGHT
) -> str:
    floor = "rough, bumpy ground" if scene.terrain == "rough" else "flat floor"
    lines = [
        f"Robot: Unitree Go1 quadruped in MuJoCo Playground (JAX/MJX), {floor}.",
        f"One policy step is {CTRL_DT} s; an episode is {EPISODE_STEPS} steps ({EPISODE_STEPS * CTRL_DT:.0f} s).",
    ]
    if not scene.is_empty:
        lines += ["", "Scene:", scene.summary()]
    lines += ["", "reward(state, action, prev_action) gets:"]
    for f in FEATURES:
        shape = "scalar" if not f.shape else f"shape {f.shape}"
        lines.append(f"- state[\"{f.name}\"] ({shape}): {f.meaning}")
    lines.append(
        "- action, prev_action (shape (12,)): the policy's joint offsets in [-1, 1]; the motor target is "
        "the standing pose + 0.5 rad * action, joint order as above"
    )
    lines.append(
        "- Body pitch (nose up positive) is arcsin(-gravity[0]) and roll (left side up) is arcsin(-gravity[1]), "
        "in radians; a foot is off the floor when its foot_contact is 0."
    )
    if scene.terrain == "rough":
        lines.append("- On rough ground, base_height and feet_height are world z, so they vary with the bumps (up to 5 cm).")
    if scene.objects:
        lines += ["", "Per object, for each object name above (e.g. state[\"ball_dist\"]):"]
        for suffix, shape, meaning in OBJECT_FEATURES + MOVABLE_FEATURES:
            only = " (movable objects only)" if (suffix, shape, meaning) in MOVABLE_FEATURES else ""
            lines.append(f"- state[\"<name>_{suffix}\"] ({'scalar' if not shape else f'shape {shape}'}){only}: {meaning}")
        target = scene.target_object
        if target is not None:
            lines.append(f"- the same keys with \"target_\" instead of \"<name>_\" (e.g. state[\"target_dist\"]) "
                         f"refer to the target object, {target.name}")
        lines.append(f"Objects present: {', '.join(o.name for o in scene.objects)}. "
                     "Use only these names; any other key raises a KeyError.")
        if scene.goal_object is not None:
            for f in GOAL_FEATURES:
                lines.append(f"- state[\"{f.name}\"] ({'scalar' if not f.shape else f'shape {f.shape}'}): {f.meaning}")
        if target is not None:
            lines.append(f"The base is within reach of {target.name} when {target.name}_dist < "
                         f"{target.reach_radius:.2f} m.")
        if scene.goal_object is not None:
            lines.append(f"The task is done when {target.name} is in {scene.goal}: target_to_goal_dist < "
                         f"{scene.goal_radius:.2f} m. Success is measured on the object, not on the robot.")
    lines += [
        "",
        "How the reward is used:",
        "- Return a dict of named scalar terms. Each step the terms are summed, multiplied by "
        f"{CTRL_DT}, and clipped below at 0. A step whose terms sum to a negative number gives 0, "
        "so keep a positive task term that outweighs the penalties (exp(-error^2 / sigma) kernels work well).",
        "- The episode ends early only if the robot flips upside down (up_z < 0).",
        "- The robot starts standing, with a random heading and small random velocities.",
        "- There is no velocity command. Reward the behavior directly.",
    ]
    if warm_start:
        lines.append(
            "- The policy starts from a pretrained walker (Playground's Go1 joystick policy), not from random "
            "weights: at the first step it already stands steadily and has a clean trot, but its command is 0, "
            "so it stands still until your terms reward motion. Your reward teaches the task; it doesn't have to "
            "teach walking or balance."
        )
    if scene.objects:
        lines += [
            "- Objects are placed fresh each episode (ranges above), and the policy observes each object's "
            "relative position, so reward the skill (e.g. distance or heading to the object), not a fixed spot.",
            "- The feet and the body (trunk) collide with solid objects; the thighs and calves pass through them.",
            "- The foot_contact feature counts touching an object as contact.",
        ]
        if any(o.kind == "zone" for o in scene.objects):
            lines.append("- Zones are painted on the floor; nothing collides with them.")
    lines += stock_lines(stock_profile, stock_weight)
    return "\n".join(lines)
