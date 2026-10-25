"""Scene gate: check a scene spec in physics before any reward is written for it.

The scene writer's spec already passed scene.parse (shapes, ranges, limits). This
compiles it into the real Go1 model and, on the CPU with plain MuJoCo:

- draws SAMPLES episode starts with the env's own placement code and checks that
  no solid piece overlaps the robot's start or another piece,
- holds the robot in its standing pose for STEP_S seconds of physics from each
  start and checks nothing explodes (non-finite or runaway state, a launched robot,
  objects through the floor),
- times those steps against the plain floor, so a scene can't slow training down
  past STEP_BUDGET.

Problems come back as short messages for the scene writer, like the reward gate's.
static_errors() holds the checks that need no MuJoCo, for the fake simulator too.
"""

import math
import time
from dataclasses import dataclass, field

import numpy as np

from . import scene_xml
from .scene import SceneSpec

SAMPLES = 8  # episode starts per scene
STEP_S = 3.0  # seconds of physics from each start
STEP_BUDGET = 3.0  # a scene's physics step may cost at most this many plain-floor steps
ROBOT_RADIUS = 0.35  # m; the Go1's footprint around its base, standing
OVERLAP = 0.01  # m of penetration between two pieces at reset that counts as overlapping
MAX_SPEED = 30.0  # m/s; anything faster has exploded
MAX_ROBOT_Z = 1.2  # m; a robot higher than this was thrown


@dataclass
class SceneReport:
    error: str | None  # None = pass; otherwise a message for the scene writer
    stats: dict[str, float] = field(default_factory=dict)


def static_errors(spec: SceneSpec) -> list[str]:
    extra = scene_xml.extra_contacts(spec)
    if extra > scene_xml.MAX_EXTRA_CONTACTS:
        return [f"the scene needs about {extra} contacts per step, more than the {scene_xml.MAX_EXTRA_CONTACTS} "
                "the simulator can afford; use fewer or simpler pieces (fewer steps, a smaller field)"]
    return []


def check_scene(spec: SceneSpec) -> str | None:
    return run_scene_gate(spec).error


def run_scene_gate(spec: SceneSpec, samples: int = SAMPLES, step_s: float = STEP_S, seed: int = 0) -> SceneReport:
    errors = static_errors(spec)
    if errors or spec.is_empty:
        return SceneReport(_message(errors) if errors else None)

    import jax
    import jax.numpy as jnp
    import mujoco

    from . import go1
    from .run_one import WARM_UP_REWARD

    with jax.default_device(jax.devices("cpu")[0]):
        env = go1.make_env(WARM_UP_REWARD, impl="jax", scene=spec)
        m = env.mj_model
        owners = _owners(m)
        pieces = sorted(owners)
        base_qpos = np.asarray(env._init_q)
        problems: dict[str, set[str]] = {}
        step_times = []
        rng = jax.random.PRNGKey(seed)
        for i in range(samples):
            rng, key, kyaw = jax.random.split(rng, 3)
            yaw = jax.random.uniform(kyaw, minval=-3.14, maxval=3.14)
            qpos = jnp.asarray(base_qpos).at[3:7].set(jnp.stack([jnp.cos(yaw / 2), 0.0, 0.0, jnp.sin(yaw / 2)]))
            qvel = jnp.zeros(m.nv)
            qpos, qvel, mocap_pos, mocap_quat = env._place_objects(key, qpos, qvel, yaw)
            d = mujoco.MjData(m)
            d.qpos[:], d.qvel[:] = np.asarray(qpos), np.asarray(qvel)
            if m.nmocap:
                d.mocap_pos[:], d.mocap_quat[:] = np.asarray(mocap_pos), np.asarray(mocap_quat)
            d.ctrl[:] = base_qpos[7:go1.ROBOT_NQ]
            mujoco.mj_forward(m, d)
            _check_start(m, d, owners, pieces, problems)
            t0 = time.perf_counter()
            _check_steps(m, d, owners, round(step_s / m.opt.timestep), problems)
            step_times.append((time.perf_counter() - t0) / max(1, round(step_s / m.opt.timestep)))

    ratio = float(np.median(step_times)) / _plain_step_time()
    stats = {"step_ratio": round(ratio, 2), "geoms": float(spec.geom_count),
             "contacts": float(scene_xml.extra_contacts(spec))}
    errors = [f"{who}: {'; '.join(sorted(what))}" for who, what in sorted(problems.items())]
    if ratio > STEP_BUDGET:
        errors.append(f"each physics step takes {ratio:.1f}x as long as on the plain floor (at most {STEP_BUDGET:g}x); "
                      "use fewer pieces")
    return SceneReport(_message(errors) if errors else None, stats)


def _message(errors: list[str]) -> str:
    return "scene failed the physics check:\n" + "\n".join(f"- {e}" for e in errors)


def _owners(m) -> dict[int, str]:
    """geom id -> the scene object or structure it belongs to, for solid scene pieces."""
    out = {}
    for g in range(m.ngeom):
        name = m.geom(g).name
        if name.startswith(scene_xml.PREFIX) and name != scene_xml.BUMPER and (m.geom_contype[g] or m.geom_conaffinity[g]):
            body = m.body(m.geom_bodyid[g]).name
            out[g] = body[len(scene_xml.PREFIX):]
    return out


def _footprint_gap(m, d, g: int, xy: np.ndarray) -> float:
    """Horizontal distance from xy to geom g's footprint (its shadow on the floor)."""
    pos, rot, size = d.geom_xpos[g], d.geom_xmat[g].reshape(3, 3), m.geom_size[g]
    kind = m.geom_type[g]
    if kind == mujoco_types()["box"]:
        offset = np.array([xy[0] - pos[0], xy[1] - pos[1], 0.0])
        if _upright(rot):  # exact: the rectangle in the box's own frame
            gap = np.maximum(np.abs((rot.T @ offset)[:2]) - size[:2], 0.0)
        else:  # tilted (ramps): its shadow in the frame turned to its heading; exact for yaw + pitch
            h = np.arctan2(rot[1, 0], rot[0, 0])
            turn = np.array([[np.cos(h), -np.sin(h), 0.0], [np.sin(h), np.cos(h), 0.0], [0.0, 0.0, 1.0]])
            local = turn.T @ rot
            gap = np.maximum(np.abs((turn.T @ offset)[:2]) - np.abs(local[:2, :]) @ size, 0.0)
        return float(np.hypot(*gap))
    if kind == mujoco_types()["capsule"]:
        axis = rot[:, 2] * size[1]
        a, b = pos[:2] - axis[:2], pos[:2] + axis[:2]
        ab = b - a
        t = np.clip(np.dot(xy - a, ab) / max(np.dot(ab, ab), 1e-9), 0.0, 1.0)
        return float(np.linalg.norm(xy - (a + t * ab)) - size[0])
    return float(np.linalg.norm(xy - pos[:2]) - size[0])


def _upright(rot: np.ndarray) -> bool:
    return abs(rot[2, 2]) > 0.999


def mujoco_types() -> dict[str, int]:
    import mujoco

    return {"box": int(mujoco.mjtGeom.mjGEOM_BOX), "capsule": int(mujoco.mjtGeom.mjGEOM_CAPSULE),
            "sphere": int(mujoco.mjtGeom.mjGEOM_SPHERE)}


def _check_start(m, d, owners: dict[int, str], pieces: list[int], problems: dict[str, set[str]]) -> None:
    base = d.qpos[:2].copy()
    for g in pieces:
        if d.geom_xpos[g][2] - _half_height(m, d, g) > 0.5:
            continue  # floats above the standing robot (a high bar)
        gap = _footprint_gap(m, d, g, base)
        if gap < ROBOT_RADIUS:
            problems.setdefault(owners[g], set()).add(
                f"comes within {max(gap, 0.0):.2f} m of the robot's base at the start of some episodes (the robot "
                f"needs {ROBOT_RADIUS} m); increase its distance or narrow its bearing")
    import mujoco

    # Two fixed pieces never collide with each other, so ask for their distance directly.
    fromto = np.zeros(6)
    for i, a in enumerate(pieces):
        for b in pieces[i + 1:]:
            if owners[a] == owners[b]:
                continue
            if mujoco.mj_geomDistance(m, d, a, b, OVERLAP, fromto) < -OVERLAP:
                first, second = sorted((owners[a], owners[b]))
                problems.setdefault(first, set()).add(
                    f"overlaps {second} at the start of some episodes; separate their distance or bearing ranges")


def _half_height(m, d, g: int) -> float:
    rot = d.geom_xmat[g].reshape(3, 3)
    kind, size = m.geom_type[g], m.geom_size[g]
    types = mujoco_types()
    if kind == types["box"]:
        return float(np.abs(rot[2, :]) @ size)
    if kind == types["capsule"]:
        return float(size[0] + size[1] * abs(rot[2, 2]))
    return float(size[0])


def _check_steps(m, d, owners: dict[int, str], steps: int, problems: dict[str, set[str]]) -> None:
    import mujoco

    bodies = {}
    for g, name in owners.items():
        bodies.setdefault(name, int(m.geom_bodyid[g]))
    start = {name: d.xpos[b].copy() for name, b in bodies.items()}
    for _ in range(steps):
        mujoco.mj_step(m, d)
    finite = np.all(np.isfinite(d.qpos)) and np.all(np.isfinite(d.qvel))
    if not finite or np.max(np.abs(d.qvel)) > MAX_SPEED:
        problems.setdefault("physics", set()).add(
            "the simulation blew up (non-finite or runaway velocities); look for pieces that overlap or are "
            "far too light or heavy")
        return
    if d.qpos[2] > MAX_ROBOT_Z:
        problems.setdefault("robot", set()).add("the robot was thrown into the air; something presses into it at the start")
    for name, b in bodies.items():
        z = d.xpos[b][2]
        if z < -0.2:
            problems.setdefault(name, set()).add("fell through the floor")
        if math.dist(d.xpos[b][:2], start[name][:2]) > 25.0:
            problems.setdefault(name, set()).add("flew away (moved over 25 m in a few seconds)")


_PLAIN: list[float] = []


def _plain_step_time() -> float:
    """Seconds per physics step of the stock flat Go1 scene, timed once per process."""
    if _PLAIN:
        return _PLAIN[0]
    import mujoco
    from mujoco_playground._src.locomotion.go1 import base as go1_base
    from mujoco_playground._src.locomotion.go1 import go1_constants as consts

    m = mujoco.MjModel.from_xml_string(consts.FEET_ONLY_FLAT_TERRAIN_XML.read_text(), assets=go1_base.get_assets())
    m.opt.timestep = 0.004
    d = mujoco.MjData(m)
    d.qpos[:] = m.keyframe("home").qpos
    d.ctrl[:] = m.keyframe("home").qpos[7:]
    steps = round(STEP_S / m.opt.timestep)
    best = math.inf
    for _ in range(3):
        t0 = time.perf_counter()
        for _ in range(steps):
            mujoco.mj_step(m, d)
        best = min(best, (time.perf_counter() - t0) / steps)
    _PLAIN.append(best)
    return best
