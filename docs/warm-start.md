# Warm starts and checkpoints

Every candidate used to learn to walk *and* its task from random weights in 10M
steps. Now every candidate builds on a pretrained Go1 walker, in one of two ways,
and any training job can continue from any saved policy.

## The pretrained walker

`scripts/train_walker.py` trains Playground's stock `Go1JoystickFlatTerrain`
policy once, with Playground's own reward, PPO config and domain randomization:

    flock /workspace/gpu.lock env MUJOCO_GL=egl .venv/bin/python scripts/train_walker.py \
        --out /workspace/pretrained/go1_walker

On the 4090 (warp) it takes 330 s: 39 s compile, then 213M steps (brax rounds
200M up to whole eval blocks). Velocity tracking on 64 randomized robots, held
command, after a settling second:

| command | reached | fell |
|---|---|---|
| stand | 0.00 m/s | 0% |
| forward 1.0 m/s | 0.97 | 0% |
| forward 1.5 m/s | 1.44 | 0% |
| backward 0.8 m/s | -0.77 | 0% |
| left 0.5 m/s | 0.43 | 0% |
| turn 1.0 rad/s | 0.97 | 0% |
| forward 0.8 + turn 0.6 | 0.76, 0.61 | 0% |

It writes `params.pkl` (final), `params_<steps>.pkl` per eval, `walker.json`
(curve, tracking) and `walker.mp4` (a command tour). The weights stay on the pod.

### The G1 walker

    /workspace/gpu_job.sh env MUJOCO_GL=egl .venv/bin/python scripts/train_walker.py --env g1 \
        --out /workspace/pretrained/g1_walker

Playground's stock `G1JoystickFlatTerrain` reward and PPO config, on robots randomized by the G1
body's randomizer. On the 4090 (warp): 1001 s for 200M steps, 141 s of it compile (155-250k steps/s).
The eval reward climbs from -6 to 14.4; Playground's termination term goes from -100 (every eval
episode fell) to -39 at 200M, on its training conditions: random pushes up to 2 m/s every 5-10 s,
joystick commands resampled every 10 s, joint angles x U(0.5, 1.5) at reset. Steer mode has no pushes
and eases the command. The tracking table (`--eval-only params.pkl`) wasn't measured: the run was
cut after `params.pkl` was saved to free the GPU for the first G1 request.

## How a candidate uses the walker (`REWARDSMITH_MODE`)

With `REWARDSMITH_WALKER` present (default `/workspace/pretrained/go1_walker/params.pkl`):

- **steer** (default): the policy steers the frozen walker. Its action is a
  command target in [-1, 1]^3, scaled to Playground's command ranges (forward
  1.5 m/s, sideways 0.8 m/s, turn 1.2 rad/s). The command eases 20% of the way
  to the target each step (the walker learned commands held for seconds), the
  policy observes it, and the walker's deterministic policy turns it into joint
  targets (`go1.Go1Env._steer`). Exploration happens in command space, so every
  random action is a real walk somewhere. The gait is the walker's: stock terms
  are off and `describe()` tells the writer to reward outcomes, not gait.
- **tune**: the policy *is* the walker's network, fine-tuned on the reward
  under the stock gait terms (below). For poses the walker can't be commanded
  into (stand on hind legs, sit, jump).
- **off**: random weights, the reward alone. The old behavior, and what you
  get with `REWARDSMITH_WARM_START=0` or no walker file.

Why two modes: on the 4 Claude rewards of the live "move ball to a target"
request, tune left every candidate standing still (ball reached in 0% of
episodes; standing already scores well and the walker's small action noise
never finds the ball), while cold training reached it in 28-100%. Steering
explores by walking.

### Tune mode details

- **Observations.** Our env's observation is the joystick observation (command
  fixed at 0) plus scene features appended at the end (`Go1Env._scene_obs`).
  `checkpoints.fit` pads the walker for the extra inputs: normalizer mean 0 /
  std 1, and zero rows in the first layer of the policy and value networks. At
  step 0 the policy acts exactly like the walker whatever the scene features
  are (`tests/test_warm_start.py`), so it stands still (its command is 0) and
  training learns how to use the scene.
- **Value network.** Restored too by default (`REWARDSMITH_RESTORE_VALUE=0` for
  a fresh one).
- **Stock terms.** Playground's gait and stability terms (`go1_spec.STOCK_SCALES`:
  orientation, pose, feet air time, slip, clearance, torques, ...) are added under
  the writer's terms as `stock_<name>`, at `REWARDSMITH_STOCK_WEIGHT` times their
  stock scales. The tracking and stand-still terms are left out (they reward
  standing still under a zero command). Profiles: `gait` (all), `pose` (drops
  the ones that hold the body level, for hind legs / sit / jump), `off`. A reward
  can override per term with a module-level `STOCK_TERMS = {"orientation": 0.0}`;
  `go1_spec.describe()` tells the writer all this.

A steer-mode checkpoint has a 3-action policy; continue it only in steer mode
(and tune / off checkpoints only outside it).

## Checkpoints (interface for the live coach)

Every training child saves the policy at every eval:

    <out_dir>/params_<steps>.pkl     # steps since this job started; params_0.pkl = the starting policy

A file is brax PPO's `(normalizer_params, policy_params, value_params)` tuple,
pickled with numpy arrays (`rewardsmith.sim.checkpoints.save` / `load`).
`checkpoints.latest(out_dir)` returns the newest. Files are written atomically
(tmp + rename), so a watcher never reads half of one. With the round preset
that is 5 files per candidate (step 0 and 4 evals), 1.6 MB each.

### Continue any job with a new reward

Any job can start from any checkpoint, with any reward source, for any number
of steps. The env is rebuilt and recompiled (the reward changed); the step
counter starts at 0, so `steps` is how many more to train.

From the command line (one process):

    python -m rewardsmith.sim.run_one new_reward.py out_dir \
        --scene scene.json --init-params old_out_dir/params_5242880.pkl --steps 5000000

From Python, several at once (one process each, warm pool and halving as usual):

    sim.run_many([
        (new_source, out_dir_a, {"init_params": "artifacts/rid/r1c2/params_5242880.pkl", "steps": 5_000_000}),
        (other_source, out_dir_b),                       # default start: the walker, preset steps
    ], scene)

or one job in-process: `sim.run(source, out_dir, scene, init_params=path, steps=n)`.

The job options are `init_params` (path) and `steps` (int); anything else is a
`ValueError`. A checkpoint must come from the same scene (or from the plain
walker, which is padded): `fit` raises if the env has fewer observations than
the checkpoint.

## Settings

| variable | default | |
|---|---|---|
| `REWARDSMITH_WARM_START` | 1 | build on the walker |
| `REWARDSMITH_WALKER` | /workspace/pretrained/go1_walker/params.pkl | the walker checkpoint |
| `REWARDSMITH_MODE` | steer (off without a walker) | steer / tune / off |
| `REWARDSMITH_STOCK` | gait | tune mode's stock profile: gait / pose / off |
| `REWARDSMITH_STOCK_WEIGHT` | 1.0 | multiplier on the stock scales |
| `REWARDSMITH_RESTORE_VALUE` | 1 | tune mode keeps the checkpoint's value network |
| `REWARDSMITH_STEPS` | the preset's | steps per candidate |
