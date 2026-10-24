"""MuJoCo Playground backend. Runs on the Runpod GPU box.

TODO(A), in order (gate G1 at 9 PM is steps 1 to 3 with the stock reward):
  1. Load the env: `from mujoco_playground import registry; env = registry.load(env_name)`.
     Confirm env names and config fields in the locomotion notebook.
  2. Train the stock reward with Brax PPO using the env's default PPO params
     (see the notebook). Time one run; that number sets K, rounds and steps.
  3. Roll out the trained policy, render frames (MUJOCO_GL=egl), save an mp4
     with mediapy, and keep ~10 evenly spaced PNG frames for the critic.
  4. Reward injection: subclass the env and replace its reward computation with
     the loaded module's terms (keep the env's termination and observations).
  5. describe(): list the exact state fields the reward may read, with shapes.
  6. gate(): load the module, jit it on a dummy state from env.reset, run ~50
     steps, reject NaN/inf. Return the traceback text on failure.
  7. Metrics: mean base forward/lateral velocity, yaw rate, base height, pitch,
     per-foot contact fraction, fall rate. The benchmark checks read these.
"""

from pathlib import Path

from .base import RunResult
from .reward_slot import load_reward


class PlaygroundSimulator:
    def __init__(self, env_name: str, train_steps: int | None = None):
        self.env_name = env_name
        self.train_steps = train_steps

    def describe(self) -> str:
        raise NotImplementedError("TODO(A): describe the env state fields")

    def gate(self, source: str) -> str | None:
        try:
            load_reward(source)
        except Exception as exc:
            return f"{type(exc).__name__}: {exc}"
        # TODO(A): jit + short rollout on a real env state
        return None

    def run(self, source: str, out_dir: Path) -> RunResult:
        raise NotImplementedError("TODO(A): train, render and measure")
