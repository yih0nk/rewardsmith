from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol

from .scene import SceneSpec


ROLLOUT_STEPS = 500  # 10 seconds at the Go1 control rate; shared by video and evaluations.


@dataclass
class RunResult:
    metrics: dict[str, Any]
    frames: list[bytes] = field(default_factory=list)  # PNG bytes, in time order, for the critic
    video_path: Path | None = None  # mp4 under the artifacts dir, for the board
    reward_curve: list[float] = field(default_factory=list)  # eval episode reward over training


@dataclass
class Stopped:
    """A job stopped mid-training by successive halving. metrics are from its last eval."""

    reason: str
    metrics: dict[str, Any]
    reward_curve: list[float] = field(default_factory=list)


# (source, out_dir) or (source, out_dir, {"init_params": checkpoint path, "steps": n})
Job = tuple[str, Path] | tuple[str, Path, dict[str, Any]]

# Called by run_many about twice a second with the indices of the jobs still training.
# Returns {job index: reason} for the jobs to stop now (see rewardsmith.search.Halving).
Cull = Callable[[set[int]], dict[int, str]]


class Simulator(Protocol):
    """scene (optional everywhere) is the request's scene spec; None means the plain flat floor."""

    def describe(self, scene: SceneSpec | None = None) -> str:
        """Notes for the reward writer: what `state`, `action` and `prev_action` hold."""
        ...

    def gate(self, source: str, scene: SceneSpec | None = None) -> str | None:
        """Compile and smoke-test a reward module. Return an error message, or None if it passes."""
        ...

    def run(self, source: str, out_dir: Path, scene: SceneSpec | None = None, **start: Any) -> RunResult:
        """Train a policy on this reward, roll it out, render, and measure.

        start may hold init_params (a checkpoint path to train from) and steps.
        """
        ...

    def run_many(
        self, jobs: list[Job], scene: SceneSpec | None = None, cull: Cull | None = None
    ) -> list["RunResult | Stopped | str"]:
        """Run several (source, out_dir[, start]) jobs. Same order as jobs; a failed job gives its error text.

        start is {"init_params": path, "steps": n}, either optional (see run).
        With cull, all jobs train at once and the ones it names are stopped early.
        """
        ...


def run_sequentially(
    sim: Simulator, jobs: list[Job], scene: SceneSpec | None = None
) -> list[RunResult | str]:
    """run_many for simulators that train one job at a time."""
    results: list[RunResult | str] = []
    for source, out_dir, *start in jobs:
        try:
            results.append(sim.run(source, out_dir, scene, **(start[0] if start else {})))
        except Exception as exc:
            results.append(f"{type(exc).__name__}: {exc}")
    return results
