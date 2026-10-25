"""No-GPU simulator: checks syntax and returns deterministic fake metrics."""

import ast
import hashlib
import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from .base import Cull, Job, OnDone, RunResult, Stopped, run_sequentially
from .go1_spec import describe
from .progress import ProgressWriter
from .scene import EMPTY, SceneSpec
from . import outcomes

FAKE_TOTAL_STEPS = 10_000_000  # the "round" preset's num_timesteps
FAKE_EVALS = 20
FAKE_COMPILE_S = 45.0  # roughly what the real first eval waits on JIT
FAKE_EVAL_S = 9.0
# CPython 3.11.5's parser can fail with "AST constructor recursion depth mismatch" (gh-106905,
# fixed in 3.11.6) when threads parse at once or after an earlier parse error; the worker gates
# drafts in threads. So parse one thread at a time, and retry once on that error.
_PARSE_LOCK = threading.Lock()


def _parse(source: str) -> ast.Module:
    with _PARSE_LOCK:
        try:
            return ast.parse(source)
        except SystemError:
            return ast.parse(source)


class FakeSimulator:
    evaluation_spec: list[dict] | None = None

    def set_evaluation_spec(self, spec: list[dict] | None) -> None:
        self.evaluation_spec = spec

    def set_check_code(self, code: str | None) -> None:
        pass  # check:<key> metrics are faked from the spec's clause names

    def describe(self, scene: SceneSpec | None = None) -> str:
        # The real Go1 contract, so prompts written against the fake match the GPU run.
        return describe(scene or EMPTY)

    def scene_gate(self, scene: SceneSpec) -> str | None:
        """The checks that need no physics (scene_gate.static_errors)."""
        from .scene_gate import _message, static_errors

        errors = static_errors(scene)
        return _message(errors) if errors else None

    def gate(self, source: str, scene: SceneSpec | None = None) -> str | None:
        try:
            tree = _parse(source)
        except SyntaxError as exc:
            return f"SyntaxError: {exc}"
        if not any(isinstance(n, ast.FunctionDef) and n.name == "reward" for n in tree.body):
            return "reward.py must define reward(state, action, prev_action)"
        return None

    def run(self, source: str, out_dir: Path, scene: SceneSpec | None = None, **start: Any) -> RunResult:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "start.json").write_text(json.dumps({k: str(v) for k, v in start.items()}))  # for tests
        (out_dir / "reward.py").write_text(source)
        h = int(hashlib.sha256(source.encode()).hexdigest()[:8], 16)
        quality = round(3 + (h % 70) / 10, 1)  # 3.0 to 9.9
        curve = [round(quality * (i + 1) / FAKE_EVALS, 2) for i in range(FAKE_EVALS)]
        terms = _fake_terms(source)
        _write_fake_progress(out_dir, curve, terms)
        return RunResult(
            metrics={
                "fake_quality": quality,
                # the rollout metrics PlaygroundSimulator reports, so the success check has data
                "forward_velocity": round((h % 200 - 100) / 100, 2),
                "lateral_velocity": round((h // 7 % 40 - 20) / 100, 2),
                "yaw_rate": round((h // 11 % 300 - 150) / 100, 2),
                "base_height": 0.27,
                "upright": 0.98,
                "fell": 0.0,
                **terms,
                **_fake_target_metrics(scene, quality),
                **{metric: round(quality / 10, 2) for metric in outcomes.requested(self.evaluation_spec, scene or EMPTY)},
                **{c["metric"]: round(quality / 10, 2) for c in self.evaluation_spec or []
                   if c["metric"].startswith("check:")},
                "scene_fell": 0.0,
            },
            reward_curve=curve,
        )

    def run_many(
        self, jobs: list[Job], scene: SceneSpec | None = None, cull: Cull | None = None,
        on_done: OnDone | None = None,
    ) -> list[RunResult | Stopped | str]:
        if cull is None:
            return list(run_sequentially(self, jobs, scene, on_done))
        results: list[RunResult | Stopped | str] = list(run_sequentially(self, jobs, scene))
        # Every fake run is already "done", with its whole curve on disk; let the cull
        # pick its rungs from those curves, then stop whoever it names.
        live = {i for i, r in enumerate(results) if isinstance(r, RunResult)}
        deadline = time.monotonic() + 30
        while live and not getattr(cull, "finished", True) and time.monotonic() < deadline:
            for i, reason in cull(set(live)).items():
                if i in live:
                    live.discard(i)
                    r = results[i]
                    results[i] = Stopped(reason=reason, metrics=r.metrics, reward_curve=r.reward_curve)
            time.sleep(0.01)
        if on_done:  # after the cull: a run it stopped must not reach the critic
            for i, r in enumerate(results):
                on_done(i, r)
        return results


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


def _fake_target_metrics(scene: SceneSpec | None, quality: float) -> dict[str, float]:
    """The metrics scene_eval reports, scaled so better fake runs get closer."""
    target = scene.target_object if scene else None
    if target is None:
        return {}
    q = quality / 10
    start = sum(target.distance) / 2
    final = round(start * (1 - q), 2)
    out = {
        "target_reached": round(q, 2),
        "target_start_dist": round(start, 2),
        "target_min_dist": final,
        "target_final_dist": final,
        "target_final_in": round(q * q, 2),
        "target_near_frac": round(q / 2, 2),
        "target_time_to_reach": round(20 - quality, 1),
        "target_fell": 0.0,
        "final_speed": round(0.5 * (1 - q), 2),
    }
    if target.movable:
        out["object_moved"] = round(2 * q, 2)
    if scene.goal_object is not None:
        g0 = round(max(1.0, sum(scene.goal_object.distance) / 2 - start), 2)
        delivered = round(max(0.0, q - 0.5) * 2, 2)
        out.update({
            "object_goal_start_dist": g0,
            "object_goal_min_dist": round(g0 * (1 - q), 2),
            "object_goal_final_dist": round(g0 * (1 - q) + 0.1, 2),
            "object_goal_reached": delivered,
            "object_in_goal_final": delivered,
            "object_goal_progress": round(q - 0.1, 2),
            "phase_no_approach": round(1 - q, 2),
            "phase_short": round(q - delivered, 2),
            "phase_delivered": delivered,
        })
    return out


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
        tree = _parse(source)
    except SyntaxError:
        return fallback
    for node in ast.walk(tree):
        if isinstance(node, ast.Return) and isinstance(node.value, ast.Dict):
            names = [k.value for k in node.value.keys if isinstance(k, ast.Constant) and isinstance(k.value, str)]
            if names:
                return names
    return fallback
