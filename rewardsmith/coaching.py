"""The live coach inside one round: watch every candidate train, comment, stop the bad ones.

At each new eval in a running candidate's progress.json, the coach (agents/coach.py)
gets that candidate's curve so far in a background thread; training never waits on
it. Its note goes to the board as soon as it lands. A kill or revise verdict stops the
candidate, within the guard rails:

- never before eval MIN_EVAL (eval 0 is the untrained policy; eval 1 is too noisy),
- never the last candidate still training,
- never more than half the round (max_stops), so a too-eager coach can't empty it,
- every decision, acted on or not, is reported with its reason.

LiveCoach is the cull run_many polls (see sim.base.Cull). With successive halving on,
it wraps the Halving cull: halving decides first, the coach then looks at the rest.
"""

import logging
from collections.abc import Callable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .agents.base import Coach, Observation
from .sim.base import Cull
from .sim.progress import point_metrics, read_progress

log = logging.getLogger("rewardsmith.coaching")

MIN_EVAL = 2
STOP_VERDICTS = ("kill", "revise")

# What happened to a verdict: the board says so next to the note.
ACTED, TOO_EARLY, LAST_RUNNER, QUOTA, FINISHED, NOTHING = (
    "stopped", "too_early", "last_runner", "quota", "finished", "none")


@dataclass
class Call:
    """One coach call: which candidate, at which eval, and what came back."""

    idx: int
    eval_index: int
    steps: int
    observation: Observation | None = None
    error: str | None = None


# (job index, the call, what was done about it)
OnNote = Callable[[Call, str], None]
# (job index, the call, the candidate's flat metrics at that eval): a candidate was stopped
OnStop = Callable[[int, Call, dict[str, Any]], None]


class LiveCoach:
    def __init__(
        self,
        coach: Coach,
        request: str,
        jobs: Sequence[tuple[str, Path]],
        *,
        spec: list[dict[str, Any]] | None = None,
        halving: Cull | None = None,
        on_note: OnNote | None = None,
        on_stop: OnStop | None = None,
        min_eval: int = MIN_EVAL,
        max_stops: int | None = None,
        max_workers: int = 4,
    ):
        self.coach, self.request, self.spec = coach, request, spec
        self.sources = [src for src, _ in jobs]
        self.out_dirs = [out_dir for _, out_dir in jobs]
        self.halving = halving
        self.on_note = on_note or (lambda call, action: None)
        self.on_stop = on_stop or (lambda idx, call, metrics: None)
        self.min_eval = min_eval
        self.max_stops = len(jobs) // 2 if max_stops is None else max_stops
        self.calls: list[Call] = []
        self._seen = [0] * len(jobs)  # curve length last sent to the coach, per job
        self._pending: dict[int, tuple[Call, Future]] = {}  # at most one call per job at a time
        self._stopped: set[int] = set()
        self._polled = False
        self._pool = ThreadPoolExecutor(max_workers=max(1, max_workers), thread_name_prefix="coach")

    @property
    def finished(self) -> bool:
        """Nothing left to decide: for simulators that poll the cull until it is (FakeSimulator)."""
        halving_done = self.halving is None or getattr(self.halving, "finished", True)
        return self._polled and halving_done and not self._pending

    def __call__(self, running: set[int]) -> dict[int, str]:
        self._polled = True
        stop = dict(self.halving(running)) if self.halving else {}
        alive = set(running) - set(stop)
        for idx, reason in self._settle(alive).items():
            stop[idx] = reason
            alive.discard(idx)
        for idx in sorted(alive):
            self._maybe_ask(idx)
        return stop

    def close(self) -> None:
        """Drop calls still in flight: the round is over."""
        self._pool.shutdown(wait=False, cancel_futures=True)

    def stats(self) -> dict[str, Any]:
        done = [c.observation for c in self.calls if c.observation is not None]
        latencies = sorted(o.latency_s for o in done)
        return {
            "calls": len(self.calls),
            "failed": sum(1 for c in self.calls if c.error),
            "stopped": len(self._stopped),
            "median_latency_s": latencies[len(latencies) // 2] if latencies else None,
            "max_latency_s": latencies[-1] if latencies else None,
            "cost_usd": sum(o.cost_usd for o in done),
        }

    def _maybe_ask(self, idx: int) -> None:
        if idx in self._pending or idx in self._stopped:
            return
        progress = read_progress(self.out_dirs[idx]) or {}
        curve = progress.get("curve") or []
        total = int(progress.get("total_steps") or 0)
        last = int(curve[-1].get("steps") or 0) if curve else 0
        if curve and (progress.get("stage") in ("rendering", "done") or (total and last >= total)):
            # The last eval: the critic judges the finished run in a moment. The coach still gets
            # the one before it if it never saw it (a run whose evals all landed at once).
            curve = curve[:-1]
        if len(curve) <= max(1, self._seen[idx]):
            return  # nothing new, or only the untrained policy so far
        self._seen[idx] = len(curve)
        steps = int(curve[-1].get("steps") or 0)
        call = Call(idx, len(curve) - 1, steps)
        self.calls.append(call)
        future = self._pool.submit(
            self.coach.observe, self.request, self.sources[idx], curve, total, self.spec
        )
        self._pending[idx] = (call, future)

    def _settle(self, alive: set[int]) -> dict[int, str]:
        """Report every finished call; return the candidates to stop now, with reasons."""
        stop: dict[int, str] = {}
        for idx, (call, future) in list(self._pending.items()):
            if not future.done():
                continue
            del self._pending[idx]
            try:
                call.observation = future.result()
            except Exception as exc:  # a failed coach call costs one note, never the run
                log.exception("coach failed on job %d at eval %d", idx, call.eval_index)
                call.error = f"{type(exc).__name__}: {exc}"
                continue
            action = self._action(idx, call, alive - set(stop))
            if action == ACTED:
                obs = call.observation
                stop[idx] = f"Stopped by the coach at {call.steps:,} steps ({obs.verdict}): {obs.note}"
                self._stopped.add(idx)
            self.on_note(call, action)
            if action == ACTED:
                curve = (read_progress(self.out_dirs[idx]) or {}).get("curve") or []
                self.on_stop(idx, call, point_metrics(curve[call.eval_index]) if len(curve) > call.eval_index else {})
        return stop

    def _action(self, idx: int, call: Call, alive: set[int]) -> str:
        assert call.observation is not None
        if call.observation.verdict not in STOP_VERDICTS:
            return NOTHING
        if idx not in alive:
            return FINISHED
        if call.eval_index < self.min_eval:
            return TOO_EARLY
        if len(alive) <= 1:
            return LAST_RUNNER
        if len(self._stopped) >= self.max_stops:
            return QUOTA
        return ACTED
