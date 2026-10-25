"""The worker: claims a request and runs write -> gate -> train -> critique rounds.

Before round 1: the scene writer turns the request into a scene spec (objects,
terrain), unless the request came with one. The spec shapes the env notes, so
every reward is written against that scene's features.
Round 1: the writer drafts K rewards from the request.
Round 2+: the editor revises the best reward so far, using the critic's notes.

While a round trains, the live coach (coaching.py) watches each candidate's evals,
comments on the board and stops candidates that are going nowhere. A stopped
candidate's fix is drafted right away, so part of the next round is written before
this one ends. Each candidate is judged by the critic as soon as its own run ends.

A run can be paused (it stops at the next round boundary and later resumes from the
database) or cancelled (it stops at the next checkpoint). See _checkpoint.

Each request is one turn of a conversation. A follow-up turn (prepare_turn) refines
one candidate of an earlier turn: same scene and check, no scene or checker call, and
round 1 is the editor revising that candidate with its critique, the coach's notes and
the user's note. Its mode (modes.py) sets K, rounds, models and the training budget.

Every request runs on one body (sim/bodies). A new task on Auto gets one from the body
router (agents/router.py) before anything else, when there is more than one to pick
from; follow-ups run on their conversation's body (route_body).
"""

import ast
import inspect
import json
import logging
import os
import re
import sqlite3
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterator, Sequence

from . import config, db, modes
from .agents import checker as check_spec
from .agents.base import Attempt, Checker, Coach, Critic, Draft, Editor, Ranker, Scener, Writer
from .agents.editor import unified_diff
from .agents.router import cards as body_cards
from .coaching import ACTED, ADVISORY, LAST_RUNNER, QUOTA, TOO_EARLY, Call, LiveCoach
from .search import Halving
from .sim import bodies
from .sim import progress as sim_progress
from .sim import scene as scene_spec
from .sim import checkpoints
from .sim.base import RunResult, Simulator, Stopped
from .sim.scene import SceneError, SceneSpec

log = logging.getLogger("rewardsmith.worker")

GATE_RETRIES = 2
SCENE_RETRIES = 2


@dataclass
class Best:
    candidate_id: int
    source: str
    score: float
    critique: str
    metrics: dict[str, Any]
    passed: bool = True  # the success check held (True when there is no check)

    def rank(self) -> tuple[bool, float]:
        """A run that passes the check beats any that doesn't; then the higher score wins."""
        return (self.passed, self.score)


@dataclass
class FollowUp:
    """What a follow-up turn's first round starts from (prepare_turn)."""
    parent: Best  # the candidate to refine; its critique carries the coach's and the user's notes
    history: list[Attempt]  # the editor's history: every earlier turn's scored candidates
    init_params: Path | None = None  # the parent's latest policy, when the simulator can continue it


class Paused(Exception):
    """A pause was requested and the run reached a round boundary."""


class Cancelled(Exception):
    """A cancel was requested and the run reached a checkpoint."""


class Refused(Exception):
    """The scene writer judged the request impossible for this robot; nothing was trained."""

    def __init__(self, feasibility: dict[str, Any]):
        super().__init__(feasibility.get("reason") or "refused")
        self.feasibility = feasibility


def _checkpoint(conn: sqlite3.Connection, rid: str, *, round_boundary: bool) -> None:
    """Act on a pending pause or cancel. Cancel stops at any checkpoint; pause only between
    rounds, so a resumed run never has a half-finished round to pick up."""
    control = db.control_of(conn, rid)
    if control == "cancel":
        raise Cancelled
    if control == "pause" and round_boundary:
        raise Paused


def run_request(
    conn: sqlite3.Connection,
    settings: config.Settings,
    req: dict[str, Any],
    writer: Writer,
    editor: Editor,
    critic: Critic,
    sim: Simulator,
    *,
    ranker: Ranker | None = None,
    checker: Checker | None = None,
    scener: Scener | None = None,
    coach: Coach | None = None,
    follow_up: FollowUp | None = None,
) -> Best | None:
    """follow_up: round 1 revises its parent instead of writing from scratch (see prepare_turn)."""
    rid = req["id"]
    text = _task(req)  # a follow-up's task carries the user's notes
    halving = settings.halving if ranker is not None else ()
    k = halving[0] if halving else settings.candidates_per_round
    # A resumed run picks up after its last finished round, with the same scene and check.
    # ranked: every scored candidate, best first; history: the same, in the order they were scored.
    start, best, ranked, history = _resume_state(conn, rid)
    if follow_up is not None and not history:  # later rounds of a follow-up still see the earlier turns
        history = list(follow_up.history)
    warm_up = getattr(sim, "warm_up", None)
    if warm_up:  # round 1's training processes warm up during the scene and check calls too
        warm_up(k)
    scener, checker = _staged(scener, conn, rid, "scene"), _staged(checker, conn, rid, "check")
    if start > 1:
        db.log_event(conn, rid, "control", f"Resuming at round {start}")
        scene = scene_spec.parse(req["scene"]) if req.get("scene") is not None else scene_spec.EMPTY
    else:
        scene = _make_scene(conn, req, scener)
    plan = getattr(sim, "plan_request", None)
    if plan:  # per-request steps and stock terms (sim/budget.py); describe() names the stock terms
        message = plan(text, scene)
        steps = getattr(sim, "steps", None)
        if settings.max_steps and steps is not None and steps > settings.max_steps:
            sim.steps = settings.max_steps  # type: ignore[attr-defined]
            message += f"; this mode caps it at {settings.max_steps / 1e6:g}M steps"
        db.log_event(conn, rid, "plan", message)
    env_doc = sim.describe(scene)
    # The critic sees the frames; naming the objects helps it judge them.
    critic_request = text if not scene.objects else f"{text}\n\nScene: {scene.summary()}"
    # A new check needs the scene, and is written while round 1's rewards are written and
    # gated; it is settled before anything trains, so the target can't move once results come in.
    pending_spec = _start_success_spec(req, checker, scene)
    spec, check_error = (None, None) if pending_spec else _settle_spec(conn, req, checker, scene, sim)
    # Next-round drafts the editor started while a round still trained (see _EarlyDrafts).
    early = _EarlyDrafts(editor, text, env_doc, max(0, k // 2))

    # source -> out_dir of every candidate that trained, stopped ones too: their params_<steps>.pkl
    # let a revision train on from its parent's weights (_start_from), including the coach's early drafts.
    trained_dirs: dict[str, Path] = {}
    if follow_up is not None and follow_up.init_params is not None:
        # A follow-up's first round revises its target, so it trains on from the target's policy.
        trained_dirs[follow_up.parent.source] = follow_up.init_params.parent
    for n in range(start, settings.max_rounds + 1):
        try:
            _checkpoint(conn, rid, round_boundary=True)
        except (Paused, Cancelled):  # the processes warm_up started would outlive the run
            close_pool = getattr(sim, "close_warm_pool", None)
            if close_pool:
                close_pool()
            early.close()
            raise
        round_id = db.add_round(conn, rid, n)
        db.log_event(conn, rid, "round", f"Round {n} started")
        try:
            warm_up = getattr(sim, "warm_up", None)  # start the training processes while Claude writes
            if warm_up:
                warm_up(k)

            try:
                on_idea = _idea_events(conn, rid)  # each idea goes on the board while Claude still writes
                with db.stage(conn, rid, "writing", n):
                    drafts = early.collect(on_idea)  # already written while the last round trained
                    rest, on_rest = k - len(drafts), _offset(on_idea, len(drafts))
                    if rest and best is None and follow_up is not None:
                        drafts += _revise(editor, text, env_doc, [follow_up.parent], rest, follow_up.history, on_rest)
                    elif rest and best is None:
                        ideas = {"on_idea": on_rest} if _accepts(writer.write, "on_idea") else {}
                        drafts += [_as_draft(d) for d in writer.write(text, env_doc, rest, **ideas)]
                    elif rest:
                        drafts += _revise(editor, text, env_doc, ranked[: settings.carry_top], rest, history, on_rest)
            except Exception as exc:
                if best is None:  # nothing to show yet: let the worker fail the request
                    raise
                # A later round failing to draft shouldn't throw away the rounds that worked.
                log.exception("request %s: drafting round %d failed", rid, n)
                db.finish_round(conn, round_id, None)
                db.log_event(conn, rid, "error", f"Stopped before round {n}: {_describe(exc)}")
                break

            # Gate every draft first, then train the survivors together: on the GPU,
            # sim.run_many trains them side by side, which hides most of the JIT time.
            # Each draft is gated in its own thread, so one draft's writer.fix call to
            # Claude overlaps the others' gates. (The simulator may still run the GPU
            # part of its gates one at a time; PlaygroundSimulator does.)
            cids = [db.add_candidate(conn, round_id, idx, d.source, d.diff, d.idea) for idx, d in enumerate(drafts)]
            lock = threading.Lock()  # the threads share conn
            with db.stage(conn, rid, "compiling", n), ThreadPoolExecutor(max_workers=max(1, len(drafts))) as pool:
                gated = list(pool.map(
                    lambda i: _pass_gate(conn, lock, rid, i, cids[i], drafts[i].source, writer, env_doc, sim, scene),
                    range(len(drafts)),
                ))
            for d, cid, src in zip(drafts, cids, gated):
                if src is not None and src != d.source and d.diff is not None and d.parent is not None:
                    db.set_candidate(conn, cid, diff=unified_diff(d.parent, src))  # the gate's fix changed it
            ready: list[tuple[int, str, Path]] = [
                (cid, src, settings.artifacts_dir / rid / f"r{n}c{idx}")
                for idx, (cid, src) in enumerate(zip(cids, gated))
                if src is not None
            ]
            if pending_spec is not None:
                spec, check_error = _settle_spec(conn, req, checker, scene, sim, pending_spec)
                pending_spec = None
            _checkpoint(conn, rid, round_boundary=False)  # before the GPU time
            if ready:
                db.log_event(conn, rid, "train", f"Training {len(ready)} candidate(s)")
            for cid, _, _ in ready:  # so the board can tell "waiting" from "training"
                db.set_candidate(conn, cid, status="training")
            parents = {cid: d.parent for d, cid in zip(drafts, cids)}
            jobs: list[tuple[Any, ...]] = [
                (src, out_dir, *(_start_from(trained_dirs, parents[cid], best, settings.continue_steps)
                                 if settings.continue_rounds else ()))
                for cid, src, out_dir in ready
            ]
            # Each candidate is judged the moment its run ends, while the others still train.
            judge = _Judge(conn, lock, rid, n, critic, critic_request, spec, check_error, settings.artifacts_dir,
                           [cid for cid, _, _ in ready], [src for _, src, _ in ready])
            cull: Any = None
            if halving and len(ready) > halving[-1]:
                assert ranker is not None
                cull = Halving(
                    [job[1] for job in jobs], halving[1:],
                    rank=lambda snapshots: ranker.rank(text, snapshots),
                    on_event=lambda msg: _locked(lock, db.log_event, conn, rid, "halving", msg),
                    labels=[job[1].name for job in jobs],
                )
            live = None
            if coach is not None and jobs:
                live = _live_coach(conn, lock, rid, n, coach, critic_request, ready, spec if not check_error else None,
                                   cull, early if n < settings.max_rounds else None, history, settings.coach_stops)
                cull = live
            more: dict[str, Any] = {"cull": cull} if cull is not None else {}
            if _accepts(sim.run_many, "on_done"):
                more["on_done"] = judge.on_done
            # "training" covers each candidate's render too; progress.json has the per-candidate stage.
            try:
                with db.stage(conn, rid, "training", n):
                    results = sim.run_many(jobs, scene, **more) if jobs else []
                judge.finish(results)  # any job run_many didn't report as it ended
                for (_, src, out_dir), result in zip(ready, results):
                    if not isinstance(result, str):
                        trained_dirs[src] = out_dir
                if live is not None:
                    _log_coach_stats(conn, lock, rid, n, live)
                _checkpoint(conn, rid, round_boundary=False)  # before waiting on the critic's Claude calls
                trained, reviews = judge.wait()
            finally:
                if live is not None:
                    live.close()
                judge.close()

            round_best: Best | None = None
            for (cid, src, result, check), review in zip(trained, reviews):
                if isinstance(review, Exception):  # _Judge already marked it failed
                    continue
                scored = Best(cid, src, review.score, review.critique, result.metrics, check["pass"] if check else True)
                ranked.append(scored)
                history.append(Attempt(n, cids.index(cid), review.score, review.critique, req.get("turn") or 1))
                if round_best is None or scored.rank() > round_best.rank():
                    round_best = scored
            # Passing runs first, then by score. Stable: on a tie the earlier candidate stays ahead.
            ranked.sort(key=lambda b: b.rank(), reverse=True)

            db.finish_round(conn, round_id, round_best.candidate_id if round_best else None)
            if round_best and (best is None or round_best.rank() > best.rank()):
                best = round_best
                db.set_request(conn, rid, best_score=best.score, best_candidate_id=best.candidate_id)
            db.log_event(conn, rid, "round", f"Round {n} done, best score {best.score if best else 'n/a'}")

            if best and n >= settings.min_rounds and _succeeded(best, settings):
                break
        finally:
            close_pool = getattr(sim, "close_warm_pool", None)
            if close_pool:
                close_pool()

    early.close()  # drafts for a round that won't run
    db.set_request(conn, rid, succeeded=int(best is not None and _succeeded(best, settings)))
    return best


def _start_from(
    trained: dict[str, Path], parent: str | None, best: Best | None, steps: int | None = None,
) -> tuple[dict, ...]:
    """A revision's job options: train on from its parent's policy, else the best one so far,
    for `steps` more steps when given (a mode's continue budget; else the simulator's).

    The scene is fixed for the whole request, so these checkpoints fit the env as they are.
    Returns () when neither has a checkpoint: the simulator's default start.
    """
    for source in (parent, best.source if best else None):
        found = checkpoints.latest(trained[source]) if source in trained else None
        if found is not None:
            return ({"init_params": str(found), **({"steps": steps} if steps else {})},)
    return ()


class _Judge:
    """The critic for one round, one candidate at a time: each run is recorded and sent to
    the critic the moment it ends (run_many's on_done), and its score lands on the board as
    soon as the critic answers, while the rest of the round still trains."""

    def __init__(
        self, conn: sqlite3.Connection, lock: threading.Lock, rid: str, round_n: int, critic: Critic,
        request: str, spec: list[dict[str, Any]] | None, check_error: str | None, artifacts_dir: Path,
        cids: list[int], sources: list[str],
    ):
        self.conn, self.lock, self.rid, self.round_n = conn, lock, rid, round_n
        self.critic, self.request, self.spec, self.check_error = critic, request, spec, check_error
        self.artifacts_dir, self.cids, self.sources = artifacts_dir, cids, sources
        self._seen: set[int] = set()
        self._trained: dict[int, tuple[int, str, RunResult, dict[str, Any] | None]] = {}
        self._reviews: dict[int, Future] = {}
        self._stage: int | None = None
        self._pool = ThreadPoolExecutor(max_workers=max(1, len(cids)), thread_name_prefix="critic")

    def on_done(self, i: int, result: Any) -> None:
        if i in self._seen:
            return
        self._seen.add(i)
        cid = self.cids[i]
        with self.lock:
            if isinstance(result, str):  # a crashed run shouldn't stop the round
                db.set_candidate(self.conn, cid, status="failed", error=result)
                return
            if isinstance(result, Stopped):  # halving or the coach cut it short; no video, so no critic
                db.set_candidate(self.conn, cid, status="stopped", metrics=result.metrics, error=result.reason)
                return
            video = _relative(result.video_path, self.artifacts_dir)
            check = check_spec.evaluate(self.spec, result.metrics) if self.spec else None
            if check is not None and self.check_error:
                check.update({"pass": False, "verified": False, "error": self.check_error})
            db.set_candidate(self.conn, cid, status="trained", metrics=result.metrics, video_path=video, check_result=check)
            if db.control_of(self.conn, self.rid) == "cancel":
                return  # the run stops at the next checkpoint: no Claude calls for it
            if self._stage is None:
                self._stage = db.start_stage(self.conn, self.rid, "judging", self.round_n)
        self._trained[i] = (cid, self.sources[i], result, check)
        self._reviews[i] = self._pool.submit(self._review, cid, result, check)

    def finish(self, results: Sequence[Any]) -> None:
        """Record whatever on_done hasn't seen (simulators without it report only here)."""
        for i, result in enumerate(results):
            self.on_done(i, result)

    def wait(self) -> tuple[list[tuple[int, str, RunResult, dict[str, Any] | None]], list[Any]]:
        """Every trained candidate in job order, with its Review or the exception the critic raised."""
        order = sorted(self._trained)
        reviews = [self._reviews[i].result() for i in order]
        if self._stage is not None:
            with self.lock:
                db.end_stage(self.conn, self._stage)
            self._stage = None
        return [self._trained[i] for i in order], reviews

    def close(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)
        if self._stage is not None:  # an error or a cancel left the stage open
            with self.lock:
                db.end_stage(self.conn, self._stage)
            self._stage = None

    def _review(self, cid: int, result: RunResult, check: dict[str, Any] | None) -> Any:
        review = _safe_review(self.critic, self.request, result, check)
        with self.lock:
            if isinstance(review, Exception):  # one failed API call shouldn't cost the rest of the run
                error = f"The critic could not score this run: {_describe(review)}"
                db.set_candidate(self.conn, cid, status="failed", error=error)
            else:
                db.set_candidate(
                    self.conn, cid, status="scored", score=review.score, critique=review.critique,
                    reward_hacking=int(getattr(review, "reward_hacking", False)),
                )
        return review


class _EarlyDrafts:
    """Next-round drafts started mid-round: when the coach stops a candidate, the editor
    revises its reward with the coach's diagnosis right away, so those drafts are ready
    when the round ends. At most `limit` a round, so the round's best still get revised."""

    def __init__(self, editor: Editor, text: str, env_doc: str, limit: int):
        self.editor, self.text, self.env_doc, self.limit = editor, text, env_doc, limit
        self._futures: list[Future] = []
        self._pool = ThreadPoolExecutor(max_workers=max(1, limit), thread_name_prefix="early-draft")

    def start(self, source: str, critique: str, metrics: dict[str, Any], history: Sequence[Attempt]) -> bool:
        """Start one revision of source. False when this round already has its share."""
        if len(self._futures) >= self.limit:
            return False
        extra = {"history": list(history)} if history and _accepts(self.editor.revise, "history") else {}

        def draft() -> list[Draft]:
            drafts = self.editor.revise(self.text, self.env_doc, source, critique, metrics, 1, **extra)
            return [replace(_as_draft(d), parent=source) for d in drafts[:1]]

        self._futures.append(self._pool.submit(draft))
        return True

    def collect(self, on_idea: Any = None) -> list[Draft]:
        """The finished drafts (waiting for any still being written). A failed one is skipped."""
        futures, self._futures = self._futures, []
        drafts: list[Draft] = []
        for future in futures:
            try:
                drafts += future.result()
            except Exception:
                log.exception("an early draft failed; the round writes that slot normally")
        if on_idea is not None:
            for i, d in enumerate(drafts):
                if d.idea:
                    on_idea(i, d.idea)
        return drafts

    def close(self) -> None:
        self._futures = []
        self._pool.shutdown(wait=False, cancel_futures=True)


def _live_coach(
    conn: sqlite3.Connection, lock: threading.Lock, rid: str, round_n: int, coach: Coach, request: str,
    ready: list[tuple[int, str, Path]], spec: list[dict[str, Any]] | None, halving: Any,
    early: "_EarlyDrafts | None", history: Sequence[Attempt], stops: bool = True,
) -> LiveCoach:
    """The round's LiveCoach, wired to the board (coach events) and to early drafts."""
    idx = [int(out_dir.name.rsplit("c", 1)[1]) for _, _, out_dir in ready]  # job -> candidate idx

    def on_note(call: Call, action: str) -> None:
        obs = call.observation
        assert obs is not None
        after = {ACTED: " Stopping it.", TOO_EARLY: " (Too early to stop it.)",
                 LAST_RUNNER: " (Kept: it is the last one still training.)",
                 QUOTA: " (Kept: half the round always finishes.)",
                 ADVISORY: " (Advice only: the coach doesn't stop runs.)"}.get(action, "")
        message = f"Candidate {idx[call.idx] + 1} at {call.steps / 1e6:.1f}M steps: {obs.note}{after}"
        _locked(lock, db.log_event, conn, rid, "coach", message, round_n=round_n, candidate=idx[call.idx])

    def on_stop(i: int, call: Call, metrics: dict[str, Any]) -> None:
        obs = call.observation
        if early is None or obs is None:
            return
        critique = (
            f"Live training coach, stopping this reward at {call.steps:,} training steps ({obs.verdict}). "
            f"What it saw: {obs.note}\nWhat to change: {obs.diagnosis}"
        )
        if early.start(ready[i][1], critique, metrics, history):
            _locked(lock, db.log_event, conn, rid, "coach",
                    f"Drafting a fix for candidate {idx[i] + 1} for round {round_n + 1} while the others train",
                    round_n=round_n, candidate=idx[i])

    return LiveCoach(coach, request, [(src, out_dir) for _, src, out_dir in ready], spec=spec, halving=halving,
                     on_note=on_note, on_stop=on_stop, stops=stops)


def _log_coach_stats(conn: sqlite3.Connection, lock: threading.Lock, rid: str, round_n: int, live: LiveCoach) -> None:
    st = live.stats()
    log.info("request %s round %d coach: %s", rid, round_n, st)
    if not st["calls"]:
        return
    latency = f", median {st['median_latency_s']:.1f} s" if st["median_latency_s"] is not None else ""
    _locked(lock, db.log_event, conn, rid, "coach",
            f"Coach: {st['calls']} looks{latency}, {st['stopped']} stopped, ${st['cost_usd']:.4f}", round_n=round_n)


def _locked(lock: threading.Lock, fn: Any, *args: Any, **kwargs: Any) -> Any:
    with lock:
        return fn(*args, **kwargs)


def _offset(on_idea: Any, by: int) -> Any:
    """on_idea for drafts numbered after the first `by`."""
    return None if on_idea is None else (lambda i, idea: on_idea(by + i, idea))


def _resume_state(
    conn: sqlite3.Connection, rid: str
) -> tuple[int, Best | None, list[Best], list[Attempt]]:
    """(first round to run, best so far, every scored candidate best first, the editor's
    history), from the rounds already in the database. A fresh request starts at round 1
    with nothing."""
    req = db.get_request(conn, rid)
    rounds = req["rounds"]
    history = [
        Attempt(r["n"], c["idx"], c["score"], c["critique"] or "", req.get("turn") or 1)
        for r in rounds for c in r["candidates"] if c["status"] == "scored"
    ]
    ranked = [
        Best(c["id"], c["reward_src"], c["score"], c["critique"] or "", c["metrics"] or {},
             (c.get("check_result") or {}).get("pass", True))
        for r in rounds for c in r["candidates"] if c["status"] == "scored"
    ]
    ranked.sort(key=lambda b: b.rank(), reverse=True)  # stable: on a tie the earlier candidate stays ahead
    return len(rounds) + 1, (ranked[0] if ranked else None), ranked, history


def _make_scene(conn: sqlite3.Connection, req: dict[str, Any], scener: Scener | None) -> SceneSpec:
    """The request's scene: the one it came with, else the scene writer's, else the plain floor.

    Rejected specs go back to the scene writer with the validator's errors, like
    the reward gate. Every attempt is logged as a "scene" event.
    """
    rid, text = req["id"], _task(req)
    if req.get("scene") is not None:
        scene = scene_spec.parse(req["scene"])
        db.log_event(conn, rid, "scene", f"Using the scene the request came with. {scene.summary()}")
        return scene
    if scener is None:
        return scene_spec.EMPTY

    try:
        spec = scener.make(text)
        for attempt in range(SCENE_RETRIES + 1):
            try:
                scene = scene_spec.parse(spec)
                break
            except SceneError as exc:
                db.log_event(conn, rid, "scene", f"Scene attempt {attempt + 1} rejected: {exc}")
                if attempt == SCENE_RETRIES:
                    db.log_event(conn, rid, "scene", "Falling back to the plain flat floor")
                    scene = scene_spec.EMPTY
                    break
                spec = scener.fix(text, spec, str(exc))
    except Exception as exc:  # the scene writer's Claude call failed: don't lose the request
        log.exception("scene writer failed for %s", rid)
        db.log_event(conn, rid, "scene", f"Could not write the scene ({_describe(exc)}); using the plain flat floor")
        scene = scene_spec.EMPTY
    db.set_request(conn, rid, scene=scene.to_dict())
    db.log_event(conn, rid, "scene", scene.summary())
    return scene


class _Staged:
    """Records a stage around every call on an agent (see db.stage).

    Used for the scene writer and the checker, so their code here stays as it is.
    """

    def __init__(self, inner: Any, conn: sqlite3.Connection, rid: str, name: str):
        self._inner, self._conn, self._rid, self._name = inner, conn, rid, name

    def __getattr__(self, attr: str) -> Any:
        fn = getattr(self._inner, attr)
        if not callable(fn):
            return fn

        def call(*args: Any, **kwargs: Any) -> Any:
            with db.stage(self._conn, self._rid, self._name):
                return fn(*args, **kwargs)

        return call


def _staged(agent: Any, conn: sqlite3.Connection, rid: str, name: str) -> Any:
    return None if agent is None else _Staged(agent, conn, rid, name)


def _safe_review(critic: Critic, text: str, result: Any, check: dict[str, Any] | None):
    """_review, but a failure comes back as the exception instead of raising: the calls run
    in a thread pool, where one raise would cost the whole round."""
    try:
        return _review(critic, text, result, check)
    except Exception as exc:
        log.exception("critic failed")
        return exc


def _describe(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


def _review(critic: Critic, text: str, result: Any, check: dict[str, Any] | None):
    """Pass the success check only when there is one, so two-argument critics keep working."""
    return critic.review(text, result, check) if check is not None else critic.review(text, result)


def _succeeded(best: Best, settings: config.Settings) -> bool:
    return best.passed and best.score >= settings.success_score


def _start_success_spec(req: dict[str, Any], checker: Checker | None, scene: SceneSpec) -> Future | None:
    """Start writing a new success check in the background, or None if there is nothing to write.
    The thread only calls Claude and validates; _success_spec stores the result."""
    if req.get("success_spec") or checker is None:
        return None
    pool = ThreadPoolExecutor(max_workers=1)
    try:
        return pool.submit(_write_spec, checker, _task(req), scene)
    finally:
        pool.shutdown(wait=False)


def _write_spec(checker: Checker, text: str, scene: SceneSpec | None) -> list[dict[str, Any]]:
    spec = checker.write_spec(text, scene=scene) if scene and not scene.is_empty else checker.write_spec(text)
    return check_spec.validate_for_scene(spec, text, scene)


def _settle_spec(
    conn: sqlite3.Connection, req: dict[str, Any], checker: Checker | None, scene: SceneSpec, sim: Simulator,
    pending: Future | None = None,
) -> tuple[list[dict[str, Any]] | None, str | None]:
    """The request's check and its error, stored, and handed to the simulator to measure."""
    spec = _success_spec(conn, req, checker, scene, pending)
    check_error = db.get_request(conn, req["id"]).get("success_check_error")
    set_spec = getattr(sim, "set_evaluation_spec", None)
    if set_spec:
        set_spec(spec if not check_error else None)
    return spec, check_error


def _success_spec(
    conn: sqlite3.Connection, req: dict[str, Any], checker: Checker | None, scene: SceneSpec | None = None,
    pending: Future | None = None,
) -> list[dict[str, Any]] | None:
    """The request's success check, stored for good before any candidate trains: pending's
    result when it was started early, else written now. If it doesn't validate, a scene with a
    goal or a fixed target gets its default check; otherwise it falls back to "don't fall over"."""
    rid = req["id"]
    if req.get("success_spec"):
        if (req.get("turn") or 1) > 1 and not req.get("scene_changed"):
            # A follow-up keeps its parent's check, validated against the parent's task: the
            # note ("get behind the ball before pushing") would read as a new task.
            return req["success_spec"]
        # Keep stored specs immutable, but do not certify an old locomotion-only
        # check as proof of a scene objective after an upgrade.
        try:
            check_spec.validate_for_scene(req["success_spec"], _task(req), scene)
        except check_spec.SpecError as exc:
            db.set_request(conn, rid, success_check_error=_describe(exc))
        return req["success_spec"]
    if checker is None:
        return None
    try:
        spec = pending.result() if pending is not None else _write_spec(checker, _task(req), scene)
    except Exception as exc:
        # A scene with a goal or a fixed target implies its check; use that when Claude's didn't
        # validate, rather than "don't fall". Not when Claude said the task can't be measured.
        default = (check_spec.default_scene_check(scene)
                   if isinstance(exc, check_spec.SpecError) and not str(exc).startswith("Unsupported") else None)
        if default is not None:
            log.warning("success check for %s rejected (%s); using the scene's default", rid, exc)
            db.log_event(conn, rid, "check", f"The written check was rejected ({exc}); using the scene's default")
            spec = default
        else:
            log.exception("success check for %s failed", rid)
            db.log_event(conn, rid, "check", f"Could not write the success check ({type(exc).__name__}); using: don't fall")
            spec = [check_spec.NO_FALL]
            db.set_request(conn, rid, success_check_error=_describe(exc))
    db.set_success_spec(conn, rid, spec)
    db.log_event(conn, rid, "check", "Success check: " + ", ".join(f"{c['metric']} {c['op']} {c['value']:g}" for c in spec))
    return spec


def _task(req: dict[str, Any]) -> str:
    """What the agents work toward: a follow-up's task (the behavior plus the user's notes), else the text."""
    return req.get("task") or req["text"]


def _as_draft(item: Draft | str | tuple[str, str | None]) -> Draft:
    """Writers may return plain sources and editors (source, diff) pairs; both become Drafts."""
    if isinstance(item, Draft):
        return _unswapped(item)
    if isinstance(item, str):
        return Draft(item)
    return Draft(item[0], item[1])


def _parses(text: str | None) -> bool:
    try:
        ast.parse(text or "")
    except SyntaxError:
        return False
    return True


def _unswapped(draft: Draft) -> Draft:
    """A draft whose idea holds the code and whose source holds the prose, put right.

    Opus 5.5 at medium effort filled every candidate this way round on a live request
    (the idea streamed as code); the gate then spent a fix call per candidate on prose.
    """
    if draft.idea and "def reward" in draft.idea and not _parses(draft.source) and _parses(draft.idea):
        return replace(draft, source=draft.idea, idea=draft.source)
    return draft


def _revise(
    editor: Editor,
    text: str,
    env_doc: str,
    parents: list[Best],
    k: int,
    history: Sequence[Attempt] = (),
    on_idea: Any = None,
) -> list[Draft]:
    """Revise each parent into its share of the k drafts, all editor calls at once.

    Revising the top two, not just the best, keeps a runner-up with a different
    idea alive: one lucky critic score shouldn't decide the whole search.
    """
    shares = [k // len(parents) + (i < k % len(parents)) for i in range(len(parents))]
    jobs = [(p, n) for p, n in zip(parents, shares) if n]
    # Older editors take no history; pass it only to those that accept it.
    extra = {"history": list(history)} if history and _accepts(editor.revise, "history") else {}

    # Draft numbers run across parents, so parent j's ideas start at the drafts before it.
    offsets = [sum(n for _, n in jobs[:j]) for j in range(len(jobs))]
    streams = on_idea is not None and _accepts(editor.revise, "on_idea")

    def revise(j: int) -> list[Draft]:
        parent, n = jobs[j]
        more = dict(extra)
        if streams:
            more["on_idea"] = lambda i, idea: on_idea(offsets[j] + i, idea)
        drafts = editor.revise(text, env_doc, parent.source, parent.critique, parent.metrics, n, **more)
        return [replace(_as_draft(d), parent=parent.source) for d in drafts]

    with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
        return [draft for drafts in pool.map(revise, range(len(jobs))) for draft in drafts]


def _idea_events(conn: sqlite3.Connection, rid: str):
    """on_idea for writers and editors: log each streamed idea as an "idea" event."""
    lock = threading.Lock()  # the editor streams from its thread pool

    def on_idea(i: int, idea: str) -> None:
        with lock:
            db.log_event(conn, rid, "idea", f"Candidate {i + 1}: {idea}")

    return on_idea


def _accepts(fn: Any, name: str) -> bool:
    try:
        return name in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False


def _pass_gate(
    conn: sqlite3.Connection,
    lock: threading.Lock,
    rid: str,
    idx: int,
    cid: int,
    src: str,
    writer: Writer,
    env_doc: str,
    sim: Simulator,
    scene: SceneSpec | None = None,
) -> str | None:
    """Return a source that compiles, asking the writer to fix errors, or None.

    Every failed attempt is logged as a "gate" event, so we can show what Claude
    got wrong and how it fixed it, even when the retry passes.
    """
    for attempt in range(GATE_RETRIES + 1):
        error = sim.gate(src, scene)
        if error is not None:
            with lock:
                # idx + 1: the board numbers candidates from 1.
                db.log_event(conn, rid, "gate", f"Candidate {idx + 1}, attempt {attempt + 1} failed: {error}")
        if error is None:
            if attempt:
                with lock:
                    db.set_candidate(conn, cid, reward_src=src)
            return src
        if attempt == GATE_RETRIES:
            break
        try:
            src = writer.fix(src, error, env_doc)
        except Exception as exc:  # a failed fix loses this candidate, not the round
            log.exception("request %s: writer.fix failed on candidate %d", rid, cid)
            error = f"{error}\n\nThe writer could not fix it: {_describe(exc)}"
            break
    with lock:
        db.set_candidate(conn, cid, status="gate_failed", error=error)
    return None


def _relative(path: Path | None, root: Path) -> str | None:
    if path is None:
        return None
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


# Notes that only ask for more of the same: the editor improves on what the critic saw.
KEEP_GOING = re.compile(r"^\s*(keep going|continue|keep training|more|again|go on|improve( it)?)[.!]*\s*$", re.I)


def _latest_params(out_dir: Path) -> Path | None:
    """A run's latest saved policy (params_<steps>.pkl), or None."""
    return checkpoints.latest(out_dir) if out_dir.is_dir() else None


def label(cand: dict[str, Any]) -> str:
    """How the board names a candidate: "Round 2 · Candidate 3", with its turn after turn 1."""
    name = f"Round {cand['round_n']} · Candidate {cand['idx'] + 1}"
    return name if (cand.get("turn") or 1) == 1 else f"Turn {cand['turn']} · {name}"


def _rank(cand: dict[str, Any]) -> tuple[bool, float, int, int, int, int]:
    """How a follow-up picks its target: a run that passes the check, then the higher score.

    Scores tie a lot (the critic caps a run that fails the task's criterion at 4), so on a tie the
    run that meets more of the check wins, then the latest: it carries the last turn's fixes and
    has a saved policy to train on from. Taking the earliest made every "keep going" revise the
    same turn-1 run again, from scratch, and drop what the turn before it had gained.
    """
    check = cand.get("check_result") or {}
    met = sum(1 for clause in check.get("clauses") or () if clause.get("pass"))
    return (check.get("pass", True), cand["score"], met, cand.get("turn") or 1, cand["round_n"], cand["idx"])


def prepare_turn(
    conn: sqlite3.Connection, settings: config.Settings, req: dict[str, Any], reader: Any = None,
) -> tuple[dict[str, Any], FollowUp | None]:
    """Set up a follow-up turn before it runs: pick the candidate it refines (the one the user
    chose, else the conversation's best), carry over that turn's scene and success check unless
    the note changes the scene, and write the turn's task. Returns the request as stored now
    and the FollowUp for run_request, or (req, None) for a first turn or a resumed one.

    reader: a follow-up reader (agents/followup.py) that says whether a note changes the scene.
    """
    rid = req["id"]
    if (req.get("turn") or 1) == 1 or db.get_request(conn, rid)["rounds"]:
        return req, None  # a first turn, or a follow-up resuming after its first round
    note = req["text"].strip()
    keep_going = not note or bool(KEEP_GOING.match(note))
    target = db.get_candidate(conn, req["target_candidate_id"]) if req.get("target_candidate_id") else None
    if target is None:
        scored = [c for c in db.scored_candidates(conn, req["conversation_id"]) if c["request_id"] != rid]
        target = max(scored, key=_rank) if scored else None
    if target is None or target["status"] != "scored":
        # Nothing to refine yet: a fresh turn on the task plus the note, with its own scene and check.
        first = db.get_request(conn, req["conversation_id"]) or req
        base = first.get("task") or first["text"]
        db.set_request(conn, rid, task=base if keep_going else f"{base}\nThe user's follow-up: {note}")
        db.log_event(conn, rid, "followup", "Nothing in this task has a score yet, so Claude writes new rewards with your note.")
        return db.get_request(conn, rid), None

    parent_req = db.get_request(conn, target["request_id"])
    base = parent_req.get("task") or parent_req["text"]
    task = base if keep_going else f"{base}\nThe user's follow-up: {note}"
    scene = scene_spec.parse(parent_req["scene"]) if parent_req.get("scene") is not None else scene_spec.EMPTY
    changed, why = False, ""
    if not keep_going and reader is not None:
        try:
            changed, why = reader.changes_scene(base, scene.summary(), note)
        except Exception as exc:  # the scene stays; a note about the world still reaches the editor
            log.exception("follow-up reader failed for %s", rid)
            db.log_event(conn, rid, "followup", f"Could not tell whether your note changes the scene ({_describe(exc)}); keeping it.")
    fields: dict[str, Any] = {"task": task, "target_candidate_id": target["id"], "scene_changed": int(changed)}
    if not changed:
        fields["scene"] = scene.to_dict()
        fields["success_check_error"] = parent_req.get("success_check_error")
    db.set_request(conn, rid, **fields)
    if not changed and parent_req.get("success_spec"):
        db.set_success_spec(conn, rid, parent_req["success_spec"])

    init_params = None
    if not changed:
        init_params = _latest_params(
            sim_progress.candidate_dir(settings.artifacts_dir, target["request_id"], target["round_n"], target["idx"])
        )
    what = "Keep going" if keep_going else f"“{note}”"
    db.log_event(conn, rid, "followup", f"Refining {label(target)} (score {target['score']:g}). {what}.")
    if changed:
        db.log_event(conn, rid, "followup", f"Your note changes the scene: {why} Writing a new scene and success check; "
                                            "training starts from scratch.")
    else:
        db.log_event(conn, rid, "followup", "Same scene and success check as before. " + (
            f"Training continues from its policy ({init_params.name})." if init_params is not None
            else "Training starts from scratch: that run saved no policy to train on from."))

    coach = db.coach_notes(conn, target["request_id"], target["round_n"], target["idx"])
    critique = target["critique"] or ""
    if coach:
        critique += "\n\nThe live training coach's notes on this run:\n" + "\n".join(f"- {c}" for c in coach)
    critique += (
        "\n\nThe user asked to keep going: fix what the critic saw and keep what already works." if keep_going
        else f"\n\nThe user watched this run and asks: “{note}”. Do what they ask first, and keep what already works."
    )
    parent = Best(target["id"], target["reward_src"], target["score"], critique, target["metrics"] or {},
                  (target.get("check_result") or {}).get("pass", True))
    # Every turn's attempts so far, oldest first, so the editor doesn't retry what an earlier turn tried.
    history = [
        Attempt(c["round_n"], c["idx"], c["score"], c["critique"] or "", c.get("turn") or 1)
        for c in db.scored_candidates(conn, req["conversation_id"]) if c["request_id"] != rid
    ]
    return db.get_request(conn, rid), FollowUp(parent, history, init_params)


@contextmanager
def training_budget(sim: Simulator, mode: modes.Mode) -> Iterator[None]:
    """Apply a mode's step budget and robustness scoring to the simulator for one run, and put
    them back after. A simulator with a per-request plan (sim.steps) is capped in run_request
    (settings.max_steps); one without gets the budget through the PPO override every training
    process reads (REWARDSMITH_PPO_OVERRIDES)."""
    saved: dict[str, Any] = {}
    key, env_before = "REWARDSMITH_PPO_OVERRIDES", os.environ.get("REWARDSMITH_PPO_OVERRIDES")
    if hasattr(sim, "robustness") and not mode.robustness:
        saved["robustness"] = sim.robustness
        sim.robustness = False  # type: ignore[attr-defined]
    if hasattr(sim, "steps"):
        saved["steps"] = sim.steps  # plan_request and the cap change it per request
    elif mode.steps and hasattr(sim, "preset"):
        os.environ[key] = json.dumps({**json.loads(env_before or "{}"), "num_timesteps": mode.steps})
    try:
        yield
    finally:
        for attr, value in saved.items():
            setattr(sim, attr, value)
        if env_before is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = env_before


def describe_run(settings: config.Settings, req: dict[str, Any], coach: Coach | None) -> str:
    """One line for the board: the message's mode, model and search size."""
    mode = modes.of(req)
    steps = f", {mode.steps / 1e6:g}M steps each" if mode.steps else ""
    rounds = (f"{settings.max_rounds} round" if settings.max_rounds == 1
              else f"{settings.min_rounds}–{settings.max_rounds} rounds")
    return (f"{mode.name.capitalize()} · {settings.model}: {settings.candidates_per_round} reward(s) a round, "
            f"{rounds}{steps}, live coach {'on' if coach is not None else 'off'}")


def build_router(settings: config.Settings) -> Any:
    """Picks a body for a request on Auto (agents/router.py)."""
    if settings.fake_llm:
        from .agents.router import FakeRouter

        return FakeRouter()
    import anthropic

    from .agents.router import ClaudeRouter

    return ClaudeRouter(anthropic.Anthropic(), settings.model)


def route_body(conn: sqlite3.Connection, settings: config.Settings, req: dict[str, Any], router: Any = None) -> str:
    """The request's body, stored on it: the one the user picked, the conversation's for a
    follow-up, else the router's pick for the task. With one body registered there is
    nothing to pick: it is used without a Claude call. Raises Refused when no body can
    do the task."""
    rid = req["id"]
    body = req.get("body") or ((req.get("turn") or 1) > 1 and db.conversation_body(conn, req["conversation_id"])) or None
    if body is None and len(bodies.BODIES) > 1:
        try:
            with db.stage(conn, rid, "body"):
                pick = (router or build_router(settings)).pick(_task(req), body_cards(bodies.BODIES))
        except Exception as exc:  # the router's Claude call failed: don't lose the request
            log.exception("body router failed for %s", rid)
            pick = {"body": bodies.DEFAULT_BODY, "reason": f"couldn't choose a body ({_describe(exc)}), so the default"}
        if pick["body"] is None:
            reason = pick["reason"] or "No available robot can do this"
            db.set_request(conn, rid, body_reason=reason)
            db.log_event(conn, rid, "refusal", f"Can't train this: {reason}."
                         + (f" Try instead: {pick['suggestion']}" if pick.get("suggestion") else ""))
            raise Refused({"verdict": "refuse", "reason": f"{reason}.", "simplification": "",
                           "suggestion": pick.get("suggestion", "")})
        body = pick["body"]
        db.set_request(conn, rid, body=body, body_reason=pick["reason"])
        db.log_event(conn, rid, "body", f"Using the {bodies.get_body(body).title}: {pick['reason']}")
    body = body or bodies.DEFAULT_BODY
    if req.get("body") != body:
        db.set_request(conn, rid, body=body)
    return body


@contextmanager
def on_body(sim: Simulator, body: str) -> Iterator[None]:
    """Run the simulator on one body (sim/bodies) for one request, and put the last one back after."""
    before = getattr(sim, "body", bodies.DEFAULT_BODY)
    sim.body = body  # type: ignore[attr-defined]
    try:
        yield
    finally:
        sim.body = before  # type: ignore[attr-defined]


def build_scener(settings: config.Settings, body: str | None = None) -> Scener:
    if settings.fake_llm:
        from .agents.fake import FakeScener

        return FakeScener()
    import anthropic

    from .agents.scener import ClaudeScener

    return ClaudeScener(anthropic.Anthropic(), settings.model, body=body)


def build_agents(
    settings: config.Settings, effort: str = "high", body: str | None = None,
) -> tuple[Writer, Editor, Critic]:
    """The writer, editor and critic on settings.model (a request's mode and model, see modes.py),
    for one body (None: the Go1)."""
    if settings.fake_llm:
        from .agents.fake import FakeCritic, FakeEditor, FakeWriter

        return FakeWriter(), FakeEditor(), FakeCritic()
    import anthropic

    from .agents.critic import ClaudeCritic
    from .agents.editor import ClaudeEditor, DroidEditor
    from .agents.writer import ClaudeWriter

    client = anthropic.Anthropic()
    editor: Editor
    if settings.editor == "droid":
        droid_model = os.environ.get("REWARDSMITH_DROID_MODEL") or None
        editor = DroidEditor(settings.artifacts_dir / "droid", model=droid_model, body=body)
    else:
        editor = ClaudeEditor(client, settings.model, effort, body=body)
    return (ClaudeWriter(client, settings.model, effort, body=body), editor,
            ClaudeCritic(client, settings.model, effort, body=body))


def build_reader(settings: config.Settings) -> Any:
    """Reads follow-up notes for scene changes (agents/followup.py)."""
    if settings.fake_llm:
        from .agents.followup import FakeFollowupReader

        return FakeFollowupReader()
    import anthropic

    from .agents.followup import ClaudeFollowupReader

    return ClaudeFollowupReader(anthropic.Anthropic())


def build(settings: config.Settings) -> tuple[Writer, Editor, Critic, Simulator]:
    writer, editor, critic = build_agents(settings)
    if settings.fake_sim:
        from .sim.fake import FakeSimulator

        sim: Simulator = FakeSimulator()
    else:
        from .sim.playground import PlaygroundSimulator

        sim = PlaygroundSimulator()  # the robot: REWARDSMITH_BODY (default go1)
    return writer, editor, critic, sim


def build_ranker(settings: config.Settings) -> Ranker | None:
    """The mid-training ranker for successive halving, or None when halving is off."""
    if not settings.halving:
        return None
    if settings.fake_llm:
        from .agents.fake import FakeRanker

        return FakeRanker()
    import anthropic

    from .agents.ranker import ClaudeRanker

    return ClaudeRanker(anthropic.Anthropic(), settings.model)


def build_coach(settings: config.Settings, req: dict[str, Any] | None = None, body: str | None = None) -> Coach | None:
    """The live coach for a request: its own choice ("off" or a model), else the worker's setting.
    Offline (fake LLM) any choice but "off" gets the fake coach."""
    model = (req or {}).get("coach") or settings.coach
    if model == "off":
        return None
    if settings.fake_llm:
        from .agents.fake import FakeCoach

        return FakeCoach()
    import anthropic

    from .agents.coach import ClaudeCoach

    return ClaudeCoach(anthropic.Anthropic(), model, body=body)


def build_checker(settings: config.Settings, body: str | None = None) -> Checker:
    if settings.fake_llm:
        from .agents.fake import FakeChecker

        return FakeChecker()
    import anthropic

    from .agents.checker import ClaudeChecker

    return ClaudeChecker(anthropic.Anthropic(), settings.model, body=body)


def prewarm(sim: Simulator, settings: config.Settings) -> None:
    """Start the next request's training processes while the worker waits for one.

    Their imports and env build (~20 s) used to sit between Claude's reward and the code
    check; warmed while idle, the first gate runs on a ready process. Sized for the
    largest round any mode runs; run_many closes the ones a smaller round doesn't use.
    Each job gets the request's settings (steps, mode) when it is handed over, not now.
    """
    warm_up = getattr(sim, "warm_up", None)
    if warm_up:
        warm_up(modes.largest_round(settings))


def run_turn(
    conn: sqlite3.Connection, settings: config.Settings, req: dict[str, Any], sim: Simulator, *,
    ranker: Ranker | None = None, reader: Any = None, router: Any = None,
) -> Best | None:
    """One request with its own mode, model and body: per-request agents, budget and coach."""
    mode = modes.of(req)
    run_settings = modes.settings_for(settings, req)
    scene_settings = replace(settings, model=modes.scene_model(settings, req))
    body = route_body(conn, scene_settings, req, router)
    for shared in (ranker, reader):  # built once per worker: they name the robot in their prompts
        if shared is not None:
            shared.body = body
    writer, editor, critic = build_agents(run_settings, mode.effort, body)
    req, follow_up = prepare_turn(conn, run_settings, req, reader)
    # Instant runs without the coach unless the request asked for one.
    coach = build_coach(settings, req, body) if mode.coach or req.get("coach") else None
    db.log_event(conn, req["id"], "mode", describe_run(run_settings, req, coach))
    with training_budget(sim, mode), on_body(sim, body):
        return run_request(
            conn, run_settings, req, writer, editor, critic, sim,
            ranker=ranker if run_settings.halving else None,
            checker=build_checker(scene_settings, body), scener=build_scener(scene_settings, body), coach=coach,
            follow_up=follow_up,
        )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    config.load_dotenv()
    settings = config.load()
    conn = db.connect(settings.db_path)
    for rid in db.fail_interrupted(conn):
        log.warning("request %s was cut off by a worker restart; marked failed", rid)
    _, _, _, sim = build(settings)
    ranker = build_ranker(settings)
    reader = build_reader(settings)
    router = build_router(replace(settings, model=modes.SONNET))
    log.info(
        "worker up (fake_sim=%s, fake_llm=%s, editor=%s, coach=%s)",
        settings.fake_sim, settings.fake_llm, settings.editor, settings.coach,
    )

    while True:
        req = db.claim_next_request(conn)
        if req is None:
            prewarm(sim, settings)
            time.sleep(2)
            continue
        log.info("running %s: %s", req["id"], req["text"])
        try:
            best = run_turn(conn, settings, req, sim, ranker=ranker, reader=reader, router=router)
            db.set_request(conn, req["id"], status="done", control=None)  # a late pause has nothing left to pause
            db.log_event(conn, req["id"], "done", f"Finished, best score {best.score if best else 'n/a'}")
        except Paused:
            rounds_done = len(db.get_request(conn, req["id"])["rounds"])
            log.info("request %s paused after round %d", req["id"], rounds_done)
            db.finish_paused(conn, req["id"], rounds_done)
        except Cancelled:
            log.info("request %s cancelled", req["id"])
            db.finish_cancelled(conn, req["id"])
        except Refused as exc:
            log.info("request %s refused: %s", req["id"], exc)
            db.set_request(conn, req["id"], status="refused", control=None)
        except Exception as exc:
            log.exception("request %s failed", req["id"])
            db.fail_request(conn, req["id"], _describe(exc))


if __name__ == "__main__":
    main()
