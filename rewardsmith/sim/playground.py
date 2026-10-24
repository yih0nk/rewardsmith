"""MuJoCo Playground backend: the real simulator behind the worker.

Settings (env vars):
  REWARDSMITH_IMPL    jax | warp              (default jax; try warp on a CUDA GPU)
"""

import os
from pathlib import Path

from . import gate, go1, go1_spec
from .base import RunResult


class PlaygroundSimulator:
    def __init__(self, env_name: str = go1.BASE_ENV, impl: str | None = None):
        if env_name != go1.BASE_ENV:
            raise ValueError(f"only {go1.BASE_ENV} is supported so far, got {env_name}")
        self.impl = impl or os.environ.get("REWARDSMITH_IMPL", "jax")

    def describe(self) -> str:
        return go1_spec.describe()

    def gate(self, source: str) -> str | None:
        return gate.check_reward(source, impl=self.impl)

    def run(self, source: str, out_dir: Path) -> RunResult:
        raise NotImplementedError("training is wired in the next commit")
