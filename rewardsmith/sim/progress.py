"""Live training progress: one progress.json per candidate, next to its artifacts.

The training process rewrites the file as it goes; the API server reads it, so
the board can show a candidate learning before its run finishes. Every write is
atomic (temp file, then os.replace), so a reader never sees half a file.

    {"stage": "compiling" | "training" | "rendering" | "done",
     "steps": int, "total_steps": int, "fraction": float 0..1, "wall_s": float,
     "started_at": float, "stage_started_at": float, "updated_at": float,   (time.time() epochs)
     "curve": [{"steps": int, "wall_s": float, "episode_reward": float, "terms": {name: float},
                "behavior": {name: float}}]}   (behavior, when present: train.CurvePoint.behavior)

wall_s counts from started_at. While nothing else happens (the minute of JIT compile
before the first eval) a heartbeat rewrites the file every HEARTBEAT_S, so wall_s and
updated_at keep moving. For a smooth timer, the board can also count from
stage_started_at against the API's server_time.

Kept free of jax and brax imports so the server and tests can use it.
"""

import json
import logging
import math
import os
import tempfile
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

log = logging.getLogger("rewardsmith.progress")

PROGRESS_FILE = "progress.json"
STAGES = ("compiling", "training", "rendering", "done")
HEARTBEAT_S = 2.0


def candidate_dir(artifacts_dir: Path, rid: str, round_n: int, idx: int) -> Path:
    """Where a candidate's artifacts live. Matches the orchestrator's out_dir."""
    return artifacts_dir / rid / f"r{round_n}c{idx}"


class ProgressWriter:
    """Writes out_dir/progress.json. Call stage() and point() as training moves on."""

    def __init__(
        self,
        out_dir: Path,
        total_steps: int,
        clock: Callable[[], float] | None = None,
        now: Callable[[], float] = time.time,
    ):
        self.path = Path(out_dir) / PROGRESS_FILE
        self.total_steps = int(total_steps)
        start = time.monotonic()
        self._clock = clock or (lambda: time.monotonic() - start)
        self._now = now
        self.started_at = self.stage_started_at = now()
        self.stage_name = "compiling"
        self.curve: list[dict[str, Any]] = []
        self._lock = threading.Lock()  # the heartbeat thread writes too
        self._stop = threading.Event()
        self._beat: threading.Thread | None = None

    def stage(self, stage: str) -> None:
        if stage not in STAGES:
            raise ValueError(f"unknown stage {stage!r}, expected one of {STAGES}")
        with self._lock:
            if stage != self.stage_name:
                self.stage_started_at = self._now()
            self.stage_name = stage
        if stage == "done":
            self.stop_heartbeat()
        self._write()

    def start_heartbeat(self, interval_s: float = HEARTBEAT_S) -> None:
        """Rewrite the file every interval_s until stage("done") or stop_heartbeat()."""
        if self._beat is not None:
            return

        def beat() -> None:
            while not self._stop.wait(interval_s):
                self._write()

        self._beat = threading.Thread(target=beat, name="progress-heartbeat", daemon=True)
        self._beat.start()

    def stop_heartbeat(self) -> None:
        self._stop.set()
        beat, self._beat = self._beat, None
        if beat is not None and beat is not threading.current_thread():
            beat.join(timeout=5)

    def point(self, p: Any) -> None:
        """Record one eval. p is a train.CurvePoint (or anything with the same fields)."""
        point = {
            "steps": int(p.steps),
            "wall_s": round(_num(p.wall_s) or 0.0, 2),
            "episode_reward": _num(p.episode_reward),
            "terms": {str(k): _num(v) for k, v in p.terms.items()},
        }
        if getattr(p, "behavior", None):
            point["behavior"] = {str(k): _num(v) for k, v in p.behavior.items()}
        with self._lock:
            self.curve.append(point)
            if self.stage_name != "training":
                self.stage_started_at = self._now()
            self.stage_name = "training"
        self._write()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return self._snapshot()

    def _snapshot(self) -> dict[str, Any]:
        steps = self.curve[-1]["steps"] if self.curve else 0
        if self.stage_name == "done":
            fraction = 1.0
        elif self.total_steps > 0:
            fraction = min(1.0, steps / self.total_steps)  # brax rounds steps up past the target
        else:
            fraction = 0.0
        return {
            "stage": self.stage_name,
            "steps": steps,
            "total_steps": self.total_steps,
            "fraction": round(fraction, 4),
            "wall_s": round(self._clock(), 2),
            "started_at": round(self.started_at, 3),
            "stage_started_at": round(self.stage_started_at, 3),
            "updated_at": round(self._now(), 3),
            "curve": list(self.curve),
        }

    def _write(self) -> None:
        # A progress file is nice to have. Losing one must not kill a 4-minute training run.
        try:
            with self._lock:  # one writer at a time, so a stale heartbeat can't land last
                write_atomic(self.path, self._snapshot())
        except OSError:
            log.exception("could not write %s", self.path)


def write_atomic(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, allow_nan=False)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def read_progress(out_dir: Path) -> dict[str, Any] | None:
    """The parsed progress.json in out_dir, or None if it is missing or unreadable."""
    try:
        text = (Path(out_dir) / PROGRESS_FILE).read_text()
        # NaN/Infinity would make the API's JSON response fail, so turn them into null.
        data = json.loads(text, parse_constant=lambda _: None)
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def point_metrics(point: dict[str, Any]) -> dict[str, Any]:
    """A curve point from progress.json as flat metrics, named like a finished run's."""
    out: dict[str, Any] = {"steps": point.get("steps"), "episode_reward": point.get("episode_reward")}
    out.update({f"term_{k}": v for k, v in (point.get("terms") or {}).items()})
    out.update(point.get("behavior") or {})
    return out


def _num(v: Any) -> float | None:
    """A JSON-safe float: a diverged run can report NaN, which JSON can't carry."""
    f = float(v)
    return f if math.isfinite(f) else None
