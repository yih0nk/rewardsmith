# Body contract (shared by the any-robot sessions)

Goal: the Rewardsmith loop (Claude writes the scene, check and reward; GPU PPO; critic; editor;
follow-ups) works on any robot that is a `Body`. The Go1 becomes the first Body with NO
behavior change: `scripts/regression.py --compare` must stay green, and requests with no
`body` keep running exactly as today.

## Files and ownership
- `rewardsmith/sim/bodies/__init__.py`: registry. `BODIES: dict[str, Body]`, `get_body(name) -> Body`,
  `DEFAULT_BODY = "go1"`. Each body module registers itself here with a one-line import.
  Owner: the interface session. Other sessions only add their own import line.
- `rewardsmith/sim/bodies/base.py`: the `Body` protocol and `Capabilities` (below). Owner: the interface session.
- `rewardsmith/sim/bodies/go1.py`: the Go1 body. It wraps today's `sim/go1.py`; keep `sim/go1.py`
  importable as a shim so nothing else breaks. Owner: the interface session.
- `rewardsmith/sim/bodies/spot.py`, `g1.py`: the legged-bodies session.
- `rewardsmith/sim/bodies/panda.py`: the arm session.
- Routing (request `body` field, feasibility picks the body, board picker): the routing session.

## Interface (Python, in bodies/base.py)

```python
@dataclass(frozen=True)
class Capabilities:
    kind: str            # "quadruped" | "humanoid" | "arm"
    legs: int            # 4, 2, 0
    can_walk: bool
    bipedal: bool
    can_grasp: bool
    reach_m: float | None  # arms: workspace radius
    summary: str         # one line Claude reads when choosing a body, e.g.
                         # "Unitree Go1: small quadruped dog, 12 joints, walks/trots/turns, pushes objects with its body, no gripper"

class Body(Protocol):
    name: str                     # registry key: "go1", "spot", "g1", "panda"
    title: str                    # board label: "Unitree Go1"
    base_env: str                 # Playground registry name it builds on
    capabilities: Capabilities
    nq: int; nv: int              # the robot's qpos/qvel slice; scene objects are appended after it
    skill: str | None             # base-skill controller: "walker" (velocity commands), "ee" (end-effector targets), None
    skill_checkpoint: str | None  # pretrained skill params path on the pod, or None

    def make_env(self, reward_source: str, scene: SceneSpec | None = None, impl: str | None = None,
                 overrides: dict | None = None, mode: str | None = None, behavior_metrics: bool = False): ...
        # Returns an mjx_env.MjxEnv whose step() calls the generated reward(state, action, prev_action),
        # same contract as today's Go1Env.

    def describe(self, scene: SceneSpec | None = None) -> str: ...
        # Writer-facing doc: the state keys, their units, the action, what "fell" means, the stock terms.

    def metric_names(self) -> list[str]: ...   # behavior metrics the checker can use for this body
    def randomizer(self): ...                  # domain randomizer that touches only the robot slice
    def ppo_overrides(self) -> dict: ...       # on top of Playground's tuned config for base_env
    camera: dict                               # render camera settings for the 720p rollout video
```

The simulator, run_one, parallel, render, outcomes and gate take a `body` name (default "go1")
and call `get_body(body)` instead of importing Go1 symbols. Steer mode generalizes to "use the
body's `skill` controller with `skill_checkpoint`".

## Merge order and freeze
1. Interface (Go1 only, no behavior change). It rebases onto #48 (any-prompt scenes) once that merges.
2. Spot (and G1), Panda: each in its own PR, stacked on 1.
3. Routing and board picker.

Feature freeze is **5:00 AM PT**. A body that isn't trained, rendered and passing one real request by
then stays unregistered (off the board). The Go1 path must never regress.

## Testing rule (Yihan's)
Per PR: `scripts/regression.py --compare` via `/workspace/gpu_job.sh` if training code changed, plus ONE
real request exercising the PR's change. No before/after tables, no multi-run benchmarks.

## As implemented (feat/body-interface)

- `bodies/__init__.py`: `register(body)`, `get_body(name=None)` (None means `DEFAULT_BODY`), `BODIES`.
  Registration is one import line at the bottom: `from . import spot  # noqa: E402, F401`.
- **Body modules must be cheap to import**: no jax/mujoco/playground at module level. The server,
  the fake simulator and the writer prompts call `describe()` without a GPU. Import your env module
  inside `make_env()` / `randomizer()` (see `bodies/go1.py`).
- A body is a plain class instance; `Body` is a `runtime_checkable` Protocol, so
  `isinstance(b, Body)` checks the attributes exist. `skill_checkpoint` may be a property
  (the Go1's reads `REWARDSMITH_WALKER`, default `/workspace/pretrained/go1_walker/params.pkl`).
- `make_env(..., mode="steer")` drives the skill controller with `skill_checkpoint` unless
  `options["walker"]` names another checkpoint. Extra keyword `options` are body-specific
  (the Go1 takes `stock_profile`, `stock_weight`, `walker`). The simulator passes the env options
  it already has (`{"walker": path}` in steer mode, the stock terms in tune mode).
- `describe(scene, mode=..., **options)`: options are what `make_env` takes (the Go1 uses
  `stock_profile`, `stock_weight`).
- `camera`: view name -> follow-cam settings `{"azimuth", "elevation", "distance"}` (degrees, m),
  and optionally `"smooth_s"`, `"lookat"` (a fixed `[x, y, z]` instead of following the base, for arms),
  `"lookat_drop"` (m below the base to aim at, default 0.12) and `"lookat_min"` (lowest aim z, default 0.08).
  `render.py` needs a `"chase"` view; `"side"` is used for the side video when the body has one.
- Optional attribute `push_overrides: dict`: config overrides that turn on random pushes
  (robust training). Bodies without it train without pushes.
- `RewardError` lives in `bodies/base.py` (re-exported by `sim/go1.py`). Raise it from `make_env`
  for an unusable reward; the gate turns it into feedback for the writer.
- `bodies/legged.py: legged_randomizer(nq, nv, floor_geom=0, torso_body=1)` is the Go1's randomizer
  (Playground's ranges, robot slice only). Spot/G1 can reuse it if their floor is geom 0 and torso body 1.
- PPO: `train.ppo_kwargs(preset, overrides, body)` starts from Playground's tuned config for
  `body.base_env` (locomotion or manipulation registry), then our preset, then `body.ppo_overrides()`.
- The robot slice: `qpos[:body.nq]`, `qvel[:body.nv]`. Scene objects (movable ones) are appended after it;
  outcomes, robustness, the randomizer and the renderer use `nq`/`nv` and never assume 19/18.
