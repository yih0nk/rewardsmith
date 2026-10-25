"""Successive halving inside one round.

Every candidate starts training at once. At each rung (the next eval in every
run's curve) the ranker sees what each robot is doing at that point, the best
`keep` carry on, and the rest are stopped mid-training. Survivors are never
paused or restarted: their training is the same as a run nobody stopped.

    REWARDSMITH_HALVING=8,4,2   start 8, keep 4 at the first eval, 2 at the second
"""

import logging
import threading
from collections.abc import Callable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

from .sim.progress import point_metrics, read_progress

log = logging.getLogger("rewardsmith.search")

# (snapshots of the contenders, in job order) -> positions into that list, best first
Rank = Callable[[list[dict[str, Any]]], list[int]]


def parse_halving(text: str) -> tuple[int, ...]:
    """"8,4,2" -> (8, 4, 2). Empty means off. Each count must be smaller than the last."""
    counts = tuple(int(x) for x in text.replace(" ", "").split(",") if x)
    if any(b >= a or b < 1 for a, b in zip(counts, counts[1:])):
        raise ValueError(f"REWARDSMITH_HALVING must shrink at every rung, got {text!r}")
    return counts if len(counts) > 1 else ()


class Halving:
    """The cull run_many calls while the round trains (see sim.base.Cull)."""

    def __init__(self, out_dirs: Sequence[Path], keep: Sequence[int], rank: Rank,
                 on_event: Callable[[str], None] | None = None, labels: Sequence[str] | None = None):
        self.out_dirs = list(out_dirs)
        self.labels = list(labels) if labels else [str(i) for i in range(len(self.out_dirs))]
        self.keep = list(keep)  # survivors after rung r, judged at eval r + 1
        self.rank = rank
        self.on_event = on_event or (lambda msg: None)
        self.rung = 0
        self._contenders: list[int] = []
        self._steps = 0
        self._future: Future | None = None
        self._pool = ThreadPoolExecutor(max_workers=1)  # the ranker is a Claude call; don't block training
        self._lock = threading.Lock()

    @property
    def finished(self) -> bool:
        return self.rung >= len(self.keep)

    def __call__(self, running: set[int]) -> dict[int, str]:
        with self._lock:
            if self.finished:
                return {}
            if self._future is None:
                self._maybe_start(running)
                return {}
            if not self._future.done():
                return {}
            return self._decide(running)

    def _maybe_start(self, running: set[int]) -> None:
        eval_index = self.rung + 1  # curve[0] is the untrained policy
        curves = {i: (read_progress(self.out_dirs[i]) or {}).get("curve") or [] for i in running}
        if not running or any(len(c) <= eval_index for c in curves.values()):
            return  # wait until every running candidate has reached this rung's eval
        contenders = sorted(running)
        if len(contenders) <= self.keep[self.rung]:
            self.rung += 1
            return
        snapshots = [point_metrics(curves[i][eval_index]) for i in contenders]
        self._contenders, self._steps = contenders, int(snapshots[0].get("steps") or 0)
        self._future = self._pool.submit(self.rank, snapshots)

    def _decide(self, running: set[int]) -> dict[int, str]:
        future, self._future = self._future, None
        keep, contenders = self.keep[self.rung], self._contenders
        self.rung += 1
        try:
            order = _as_order(future.result(), len(contenders))
        except Exception as exc:  # a failed ranker shouldn't kill the round: keep everyone
            log.exception("ranker failed at rung %d", self.rung)
            self.on_event(f"Halving rung {self.rung}: ranker failed ({type(exc).__name__}), keeping all")
            return {}
        ranked = [contenders[p] for p in order]
        stop = {
            i: f"Stopped at {self._steps:,} steps: ranked {place + 1} of {len(ranked)}, only the top {keep} kept training"
            for place, i in enumerate(ranked) if place >= keep and i in running
        }
        kept = ", ".join(self.labels[i] for i in ranked[:keep])
        self.on_event(f"Halving rung {self.rung} at {self._steps:,} steps: kept candidates {kept} of {len(ranked)}")
        return stop


def survivor_stretch(n_jobs: int, keep: Sequence[int], num_evals: int) -> float:
    """How much longer every job of a halving round can train for the round to cost n_jobs full runs.

    Rung r is judged at eval r + 1, and evals split training into num_evals - 1 equal
    blocks, so block b runs as many jobs as survived the rungs before it. The steps
    the culled jobs never train go to the survivors, which train on from their own
    weights: 4 jobs keeping 2 at the first of 4 blocks cost (4 + 2 + 2 + 2) / 4 = 2.5
    full runs, so each job gets 4 / 2.5 = 1.6x the steps; 8,4,2 gets 2x.
    """
    blocks = max(1, num_evals - 1)
    counts = [n_jobs, *keep]
    work = sum(min(n_jobs, counts[min(b, len(keep))]) for b in range(blocks)) / blocks
    return n_jobs / work if work else 1.0


def _as_order(order: Sequence[int], n: int) -> list[int]:
    """A full ranking of range(n): the ranker's valid picks first, anything it left out after."""
    seen: list[int] = []
    for p in order:
        if isinstance(p, int) and 0 <= p < n and p not in seen:
            seen.append(p)
    return seen + [p for p in range(n) if p not in seen]
