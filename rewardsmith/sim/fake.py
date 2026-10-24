"""No-GPU simulator: checks syntax and returns deterministic fake metrics."""

import ast
import hashlib
from pathlib import Path

from .base import RunResult

ENV_DOC = """FAKE ENVIRONMENT (no GPU). Real notes come from PlaygroundSimulator.describe().
state: base linear/angular velocity, gravity in body frame, joint angles and velocities, foot contacts.
action: 12 joint targets in [-1, 1]. prev_action: the previous step's action."""


class FakeSimulator:
    def describe(self) -> str:
        return ENV_DOC

    def gate(self, source: str) -> str | None:
        try:
            tree = ast.parse(source)
        except SyntaxError as exc:
            return f"SyntaxError: {exc}"
        if not any(isinstance(n, ast.FunctionDef) and n.name == "reward" for n in tree.body):
            return "reward.py must define reward(state, action, prev_action)"
        return None

    def run(self, source: str, out_dir: Path) -> RunResult:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "reward.py").write_text(source)
        h = int(hashlib.sha256(source.encode()).hexdigest()[:8], 16)
        quality = round(3 + (h % 70) / 10, 1)  # 3.0 to 9.9
        curve = [round(quality * (i + 1) / 20, 2) for i in range(20)]
        return RunResult(
            metrics={"fake_quality": quality, "forward_velocity": round((h % 200 - 100) / 100, 2)},
            reward_curve=curve,
        )
