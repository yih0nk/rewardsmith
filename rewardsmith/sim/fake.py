"""No-GPU simulator: checks syntax and returns deterministic fake metrics."""

import ast
import hashlib
from pathlib import Path
from types import SimpleNamespace

from .base import RunResult, run_sequentially
from .go1_spec import describe
from .progress import ProgressWriter

FAKE_TOTAL_STEPS = 10_000_000  # the "round" preset's num_timesteps
FAKE_EVALS = 20
FAKE_COMPILE_S = 45.0  # roughly what the real first eval waits on JIT
FAKE_EVAL_S = 9.0


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
        curve = [round(quality * (i + 1) / FAKE_EVALS, 2) for i in range(FAKE_EVALS)]
        terms = _fake_terms(source)
        _write_fake_progress(out_dir, curve, terms)
        return RunResult(
            metrics={
                "fake_quality": quality,
                "forward_velocity": round((h % 200 - 100) / 100, 2),
                **terms,
            },
            reward_curve=curve,
        )

    def run_many(self, jobs: list[tuple[str, Path]]) -> list[RunResult | str]:
        return run_sequentially(self, jobs)


def _write_fake_progress(out_dir: Path, curve: list[float], terms: dict[str, float]) -> None:
    """Write the same progress.json a real run would, all at once, ending in "done".
    Each term ramps up to its final metric value along with the episode reward."""
    wall = [0.0]
    progress = ProgressWriter(out_dir, FAKE_TOTAL_STEPS, clock=lambda: wall[0])
    progress.stage("compiling")
    for i, reward in enumerate(curve):
        frac = (i + 1) / len(curve)
        wall[0] = round(FAKE_COMPILE_S + i * FAKE_EVAL_S, 1)
        progress.point(SimpleNamespace(
            steps=int(FAKE_TOTAL_STEPS * frac),
            wall_s=wall[0],
            episode_reward=reward,
            terms={k.removeprefix("term_"): round(v * frac, 3) for k, v in terms.items()},
        ))
    wall[0] += 6.0  # rollout and video
    progress.stage("rendering")
    progress.stage("done")


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
