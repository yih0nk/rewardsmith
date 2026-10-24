"""The worker: claims a request and runs write -> gate -> train -> critique rounds.

Before round 1: the scene writer turns the request into a scene spec (objects,
terrain), unless the request came with one. The spec shapes the env notes, so
every reward is written against that scene's features.
Round 1: the writer drafts K rewards from the request.
Round 2+: the editor revises the best reward so far, using the critic's notes.
"""

import inspect
import logging
import os
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Sequence

from . import config, db
from .agents import checker as check_spec
from .agents.base import Attempt, Checker, Critic, Draft, Editor, Ranker, Scener, Writer
from .agents.editor import unified_diff
from .search import Halving
from .sim import scene as scene_spec
from .sim.base import Simulator, Stopped
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
) -> Best | None:
    rid, text = req["id"], req["text"]
    halving = settings.halving if ranker is not None else ()
    k = halving[0] if halving else settings.candidates_per_round
    best: Best | None = None
    ranked: list[Best] = []  # every scored candidate so far, best first
    history: list[Attempt] = []  # the same, in the order they were scored, for the editor
    scene = _make_scene(conn, req, scener)
    env_doc = sim.describe(scene)
    # The critic sees the frames; naming the objects helps it judge them.
    critic_request = text if not scene.objects else f"{text}\n\nScene: {scene.summary()}"
    spec = _success_spec(conn, req, checker)
    check_error = db.get_request(conn, rid).get("success_check_error")

    for n in range(1, settings.max_rounds + 1):
        round_id = db.add_round(conn, rid, n)
        db.log_event(conn, rid, "round", f"Round {n} started")
        try:
            warm_up = getattr(sim, "warm_up", None)  # start the training processes while Claude writes
            if warm_up:
                warm_up(k)

            try:
                if best is None:
                    drafts = [_as_draft(d) for d in writer.write(text, env_doc, k)]
                else:
                    drafts = _revise(editor, text, env_doc, ranked[: settings.carry_top], k, history)
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
            with ThreadPoolExecutor(max_workers=max(1, len(drafts))) as pool:
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
            if ready:
                db.log_event(conn, rid, "train", f"Training {len(ready)} candidate(s)")
            for cid, _, _ in ready:  # so the board can tell "waiting" from "training"
                db.set_candidate(conn, cid, status="training")
            jobs = [(src, out_dir) for _, src, out_dir in ready]
            if halving and len(ready) > halving[-1]:
                assert ranker is not None
                cull = Halving(
                    [out_dir for _, out_dir in jobs], halving[1:],
                    rank=lambda snapshots: ranker.rank(text, snapshots),
                    on_event=lambda msg: db.log_event(conn, rid, "halving", msg),
                    labels=[out_dir.name for _, out_dir in jobs],
                )
                results = sim.run_many(jobs, scene, cull=cull)
            else:
                results = sim.run_many(jobs, scene) if jobs else []

            trained = []
            for (cid, src, _), result in zip(ready, results):
                if isinstance(result, str):  # a crashed run shouldn't stop the round
                    db.set_candidate(conn, cid, status="failed", error=result)
                    continue
                if isinstance(result, Stopped):  # halving cut it short; no video, so no critic
                    db.set_candidate(conn, cid, status="stopped", metrics=result.metrics, error=result.reason)
                    continue
                video = _relative(result.video_path, settings.artifacts_dir)
                check = check_spec.evaluate(spec, result.metrics) if spec else None
                if check is not None and check_error:
                    check.update({"pass": False, "verified": False, "error": check_error})
                db.set_candidate(conn, cid, status="trained", metrics=result.metrics, video_path=video, check_result=check)
                trained.append((cid, src, result, check))

            # One Claude call per video, all at once; each takes tens of seconds.
            with ThreadPoolExecutor(max_workers=max(1, len(trained))) as pool:
                reviews = list(pool.map(lambda t: _safe_review(critic, critic_request, t[2], t[3]), trained))

            round_best: Best | None = None
            for (cid, src, result, check), review in zip(trained, reviews):
                if isinstance(review, Exception):  # one failed API call shouldn't cost the rest of the run
                    error = f"The critic could not score this run: {_describe(review)}"
                    db.set_candidate(conn, cid, status="failed", error=error)
                    continue
                db.set_candidate(
                    conn,
                    cid,
                    status="scored",
                    score=review.score,
                    critique=review.critique,
                    reward_hacking=int(getattr(review, "reward_hacking", False)),
                )
                scored = Best(cid, src, review.score, review.critique, result.metrics, check["pass"] if check else True)
                ranked.append(scored)
                history.append(Attempt(n, cids.index(cid), review.score, review.critique))
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

    db.set_request(conn, rid, succeeded=int(best is not None and _succeeded(best, settings)))
    return best


def _make_scene(conn: sqlite3.Connection, req: dict[str, Any], scener: Scener | None) -> SceneSpec:
    """The request's scene: the one it came with, else the scene writer's, else the plain floor.

    Rejected specs go back to the scene writer with the validator's errors, like
    the reward gate. Every attempt is logged as a "scene" event.
    """
    rid, text = req["id"], req["text"]
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


def _success_spec(
    conn: sqlite3.Connection, req: dict[str, Any], checker: Checker | None
) -> list[dict[str, Any]] | None:
    """The request's success check, written now, before any candidate trains, and stored
    for good. Falls back to "don't fall over" if the checker fails."""
    if req.get("success_spec") or checker is None:
        return req.get("success_spec")
    rid = req["id"]
    try:
        spec = checker.write_spec(req["text"])
    except Exception as exc:
        log.exception("success check for %s failed", rid)
        db.log_event(conn, rid, "check", f"Could not write the success check ({type(exc).__name__}); using: don't fall")
        spec = [check_spec.NO_FALL]
        db.set_request(conn, rid, success_check_error=_describe(exc))
    db.set_success_spec(conn, rid, spec)
    db.log_event(conn, rid, "check", "Success check: " + ", ".join(f"{c['metric']} {c['op']} {c['value']:g}" for c in spec))
    return spec


def _as_draft(item: Draft | str | tuple[str, str | None]) -> Draft:
    """Writers may return plain sources and editors (source, diff) pairs; both become Drafts."""
    if isinstance(item, Draft):
        return item
    if isinstance(item, str):
        return Draft(item)
    return Draft(item[0], item[1])


def _revise(
    editor: Editor, text: str, env_doc: str, parents: list[Best], k: int, history: Sequence[Attempt] = ()
) -> list[Draft]:
    """Revise each parent into its share of the k drafts, all editor calls at once.

    Revising the top two, not just the best, keeps a runner-up with a different
    idea alive: one lucky critic score shouldn't decide the whole search.
    """
    shares = [k // len(parents) + (i < k % len(parents)) for i in range(len(parents))]
    jobs = [(p, n) for p, n in zip(parents, shares) if n]
    # Older editors take no history; pass it only to those that accept it.
    extra = {"history": list(history)} if history and _accepts(editor.revise, "history") else {}

    def revise(job: tuple[Best, int]) -> list[Draft]:
        parent, n = job
        drafts = editor.revise(text, env_doc, parent.source, parent.critique, parent.metrics, n, **extra)
        return [replace(_as_draft(d), parent=parent.source) for d in drafts]

    with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
        return [draft for drafts in pool.map(revise, jobs) for draft in drafts]


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


def build_scener(settings: config.Settings) -> Scener:
    if settings.fake_llm:
        from .agents.fake import FakeScener

        return FakeScener()
    import anthropic

    from .agents.scener import ClaudeScener

    return ClaudeScener(anthropic.Anthropic(), settings.model)


def build(settings: config.Settings) -> tuple[Writer, Editor, Critic, Simulator]:
    if settings.fake_llm:
        from .agents.fake import FakeCritic, FakeEditor, FakeWriter

        writer, editor, critic = FakeWriter(), FakeEditor(), FakeCritic()
    else:
        import anthropic

        from .agents.critic import ClaudeCritic
        from .agents.editor import ClaudeEditor, DroidEditor
        from .agents.writer import ClaudeWriter

        client = anthropic.Anthropic()
        writer = ClaudeWriter(client, settings.model)
        if settings.editor == "droid":
            droid_model = os.environ.get("REWARDSMITH_DROID_MODEL") or None
            editor = DroidEditor(settings.artifacts_dir / "droid", model=droid_model)
        else:
            editor = ClaudeEditor(client, settings.model)
        critic = ClaudeCritic(client, settings.model)

    if settings.fake_sim:
        from .sim.fake import FakeSimulator

        sim: Simulator = FakeSimulator()
    else:
        from .sim.playground import PlaygroundSimulator

        sim = PlaygroundSimulator(settings.env_name)
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


def build_checker(settings: config.Settings) -> Checker:
    if settings.fake_llm:
        from .agents.fake import FakeChecker

        return FakeChecker()
    import anthropic

    from .agents.checker import ClaudeChecker

    return ClaudeChecker(anthropic.Anthropic(), settings.model)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    config.load_dotenv()
    settings = config.load()
    conn = db.connect(settings.db_path)
    for rid in db.fail_interrupted(conn):
        log.warning("request %s was cut off by a worker restart; marked failed", rid)
    writer, editor, critic, sim = build(settings)
    ranker = build_ranker(settings)
    checker = build_checker(settings)
    scener = build_scener(settings)
    log.info(
        "worker up (fake_sim=%s, fake_llm=%s, editor=%s)", settings.fake_sim, settings.fake_llm, type(editor).__name__
    )

    while True:
        req = db.claim_next_request(conn)
        if req is None:
            time.sleep(2)
            continue
        log.info("running %s: %s", req["id"], req["text"])
        try:
            best = run_request(
                conn, settings, req, writer, editor, critic, sim, ranker=ranker, checker=checker, scener=scener
            )
            db.set_request(conn, req["id"], status="done")
            db.log_event(conn, req["id"], "done", f"Finished, best score {best.score if best else 'n/a'}")
        except Exception as exc:
            log.exception("request %s failed", req["id"])
            db.fail_request(conn, req["id"], _describe(exc))


if __name__ == "__main__":
    main()
