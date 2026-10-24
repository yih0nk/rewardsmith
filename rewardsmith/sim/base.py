from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


@dataclass
class RunResult:
    metrics: dict[str, Any]
    frames: list[bytes] = field(default_factory=list)  # PNG bytes, in time order, for the critic
    video_path: Path | None = None  # mp4 under the artifacts dir, for the board
    reward_curve: list[float] = field(default_factory=list)  # eval episode reward over training


class Simulator(Protocol):
    def describe(self) -> str:
        """Notes for the reward writer: what `state`, `action` and `prev_action` hold."""
        ...

    def gate(self, source: str) -> str | None:
        """Compile and smoke-test a reward module. Return an error message, or None if it passes."""
        ...

    def run(self, source: str, out_dir: Path) -> RunResult:
        """Train a policy on this reward, roll it out, render, and measure."""
        ...
