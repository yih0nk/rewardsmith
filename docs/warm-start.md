# Warm starts and checkpoints

Every candidate used to learn to walk *and* its task from random weights in 10M
steps. Now it starts from a pretrained Go1 walker, keeps Playground's gait terms
under its own reward, and any training job can continue from any saved policy.

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

## How a candidate starts

`PlaygroundSimulator` starts every job from `REWARDSMITH_WALKER` (default
`/workspace/pretrained/go1_walker/params.pkl`) when the file exists.

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

`REWARDSMITH_WARM_START=0` turns all of it off (random weights, no stock terms):
the old behavior.

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
| `REWARDSMITH_WARM_START` | 1 | start from the walker |
| `REWARDSMITH_WALKER` | /workspace/pretrained/go1_walker/params.pkl | the walker checkpoint |
| `REWARDSMITH_STOCK` | gait (off without a walker) | gait / pose / off |
| `REWARDSMITH_STOCK_WEIGHT` | 1.0 | multiplier on the stock scales |
| `REWARDSMITH_RESTORE_VALUE` | 1 | keep the checkpoint's value network |
| `REWARDSMITH_STEPS` | the preset's | steps per candidate |
