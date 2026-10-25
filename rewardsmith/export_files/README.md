# Rewardsmith policy export

Task: {{TASK}}

A Unitree Go1 policy trained by [Rewardsmith](https://github.com/yih0nk/rewardsmith): Claude wrote
the reward in `task/reward.py`, brax PPO trained it in MuJoCo Playground, and Claude's critic
scored the result (`task/critique.json`). `manifest.json` says which candidate this is.

## Run it

```bash
pip install mujoco brax playground
python run_policy.py
```

`playground` is MuJoCo Playground on PyPI; it imports as `mujoco_playground`. On first run it
downloads the MuJoCo Menagerie (the Go1 model), so it needs git and a network connection once.

That rolls the policy out for 500 control steps (10 s) on the scene it trained in, then prints
its speed, height, uprightness and falls next to the board's numbers for the same run.

- `--video out.mp4` renders the rollout (needs ffmpeg).
- `--viewer` replays it in MuJoCo's viewer. On macOS run it with `mjpython run_policy.py --viewer`.
- `--deterministic` takes the policy's mean action instead of sampling it.
- `--seed N` picks another episode; 0 is the one in `rollout.mp4`.
- `--impl jax|warp` picks MJX's backend (default: warp on a GPU, jax elsewhere).

The board trains on a GPU with the warp backend, so on a CPU the numbers come out close to the
board's, not identical.

## What's here

| Path | What |
| --- | --- |
| `policy/params.pkl` | brax PPO params `(normalizer, policy, value)`, every array as numpy |
| `policy/network.json` | the network config: hidden layers, observation keys, action size, mode |
| `policy/walker.pkl` | steer mode only: the frozen pretrained walker the policy commands |
| `task/reward.py` | the reward it trained on |
| `task/scene.json` | the scene spec (`null` is the plain flat floor) |
| `task/success_check.json` | the task's pass/fail check and this run's result |
| `task/metrics.json` | every metric the board shows for this run |
| `task/critique.json` | the critic's score and notes |
| `rollout.mp4` | the board's video of this policy |
| `rewardsmith_env/` | a copy of Rewardsmith's Go1 env, so the policy sees the observations it trained on |

## Modes

- `joints`: the policy outputs 12 joint position targets, like Playground's Go1 joystick policy.
- `steer`: the policy outputs a velocity command (forward, sideways, turn rate) in [-1, 1] each
  step. The env scales and smooths it, writes it into the walker's command input, and the walker
  (`policy/walker.pkl`, a stock Go1 joystick policy) turns it into joint targets. To run it on
  your own stack you need both networks.

## Using the params elsewhere

```python
import pickle
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks as ppo_networks

params = pickle.load(open("policy/params.pkl", "rb"))
sizes = {k: (v.shape[-1],) for k, v in params[0].mean.items()}
nets = ppo_networks.make_ppo_networks(
    sizes, action_size, preprocess_observations_fn=running_statistics.normalize,
    policy_hidden_layer_sizes=(512, 256, 128), value_hidden_layer_sizes=(512, 256, 128),
    policy_obs_key="state", value_obs_key="privileged_state",
)
policy = ppo_networks.make_inference_fn(nets)(params, deterministic=True)
```

`action_size` and the layer sizes are in `policy/network.json`. The policy reads `obs["state"]`:
Playground's 48 Go1 joystick inputs, then the scene's inputs (see `rewardsmith_env/go1.py`).
