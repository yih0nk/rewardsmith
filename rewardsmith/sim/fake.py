"""No-GPU simulator: checks syntax and returns deterministic fake metrics."""

import ast
import hashlib
from pathlib import Path

from .base import RunResult, run_sequentially
from .go1_spec import describe


class FakeSimulator:
    def describe(self) -> str:
        # The real Go1 contract, so prompts written against the fake match the GPU run.
        return describe()

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
            metrics={
                "fake_quality": quality,
                "forward_velocity": round((h % 200 - 100) / 100, 2),
                **_fake_terms(source),
            },
            reward_curve=curve,
        )

    def run_many(self, jobs: list[tuple[str, Path]]) -> list[RunResult | str]:
        return run_sequentially(self, jobs)


def _fake_terms(source: str) -> dict[str, float]:
    """One term_<name> metric per reward term, like PlaygroundSimulator, so the board's
    reward breakdown has data offline. Values are deterministic, from -1.0 to 2.0."""
    out = {}
    for name in _term_names(source):
        h = int(hashlib.sha256(f"{name}:{source}".encode()).hexdigest()[:8], 16)
        out[f"term_{name}"] = round((h % 300 - 100) / 100, 3)
    return out


def _term_names(source: str) -> list[str]:
    """String keys of the first dict literal the reward returns, or task and energy."""
    fallback = ["task", "energy"]
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return fallback
    for node in ast.walk(tree):
        if isinstance(node, ast.Return) and isinstance(node.value, ast.Dict):
            names = [k.value for k in node.value.keys if isinstance(k, ast.Constant) and isinstance(k.value, str)]
            if names:
                return names
    return fallback
