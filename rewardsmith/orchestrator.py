"""The worker: claims a request and runs write -> gate -> train -> critique rounds.

Round 1: the writer drafts K rewards from the request.
Round 2+: the editor revises the best reward so far, using the critic's notes.
"""

import logging
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import config, db
from .agents.base import Critic, Editor, Ranker, Writer
from .search import Halving
from .sim.base import Simulator, Stopped

log = logging.getLogger("rewardsmith.worker")

GATE_RETRIES = 2


@dataclass
class Best:
    candidate_id: int
    source: str
    score: float
    critique: str
    metrics: dict[str, Any]


def run_request(
    conn: sqlite3.Connection,
    settings: config.Settings,
    req: dict[str, Any],
    writer: Writer,
    editor: Editor,
    critic: Critic,
    sim: Simulator,
    ranker: Ranker | None = None,
) -> Best | None:
    rid, text = req["id"], req["text"]
    halving = settings.halving if ranker is not None else ()
    k = halving[0] if halving else settings.candidates_per_round
    best: Best | None = None
    ranked: list[Best] = []  # every scored candidate so far, best first
    env_doc = sim.describe()

    for n in range(1, settings.max_rounds + 1):
        round_id = db.add_round(conn, rid, n)
        db.log_event(conn, rid, "round", f"Round {n} started")
        warm_up = getattr(sim, "warm_up", None)  # start the training processes while Claude writes
        if warm_up:
            warm_up(k)

        if best is None:
            drafts = [(src, None) for src in writer.write(text, env_doc, k)]
        else:
            drafts = _revise(editor, text, env_doc, ranked[: settings.carry_top], k)

        # Gate every draft first, then train the survivors together: on the GPU,
        # sim.run_many trains them side by side, which hides most of the JIT time.
        # Each draft is gated in its own thread, so one draft's writer.fix call to
        # Claude overlaps the others' gates. (The simulator may still run the GPU
        # part of its gates one at a time; PlaygroundSimulator does.)
        cids = [db.add_candidate(conn, round_id, idx, src, diff) for idx, (src, diff) in enumerate(drafts)]
        lock = threading.Lock()  # the threads share conn
        with ThreadPoolExecutor(max_workers=max(1, len(drafts))) as pool:
            gated = list(pool.map(
                lambda i: _pass_gate(conn, lock, rid, i, cids[i], drafts[i][0], writer, env_doc, sim),
                range(len(drafts)),
            ))
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
            results = sim.run_many(jobs, cull=cull)
        else:
            results = sim.run_many(jobs) if jobs else []

        trained = []
        for (cid, src, _), result in zip(ready, results):
            if isinstance(result, str):  # a crashed run shouldn't stop the round
                db.set_candidate(conn, cid, status="failed", error=result)
                continue
            if isinstance(result, Stopped):  # halving cut it short; no video, so no critic
                db.set_candidate(conn, cid, status="stopped", metrics=result.metrics, error=result.reason)
                continue
            video = _relative(result.video_path, settings.artifacts_dir)
            db.set_candidate(conn, cid, status="trained", metrics=result.metrics, video_path=video)
            trained.append((cid, src, result))

        # One Claude call per video, all at once; each takes tens of seconds.
        with ThreadPoolExecutor(max_workers=max(1, len(trained))) as pool:
            reviews = list(pool.map(lambda t: critic.review(text, t[2]), trained))

        round_best: Best | None = None
        for (cid, src, result), review in zip(trained, reviews):
            db.set_candidate(conn, cid, status="scored", score=review.score, critique=review.critique)
            scored = Best(cid, src, review.score, review.critique, result.metrics)
            ranked.append(scored)
            if round_best is None or review.score > round_best.score:
                round_best = scored
        ranked.sort(key=lambda b: -b.score)  # stable: on a tie the earlier candidate stays ahead

        db.finish_round(conn, round_id, round_best.candidate_id if round_best else None)
        if round_best and (best is None or round_best.score > best.score):
            best = round_best
            db.set_request(conn, rid, best_score=best.score)
        db.log_event(conn, rid, "round", f"Round {n} done, best score {best.score if best else 'n/a'}")

        if best and n >= settings.min_rounds and best.score >= settings.success_score:
            break

    return best


def _revise(editor: Editor, text: str, env_doc: str, parents: list[Best], k: int) -> list[tuple[str, str]]:
    """Revise each parent into its share of the k drafts, all editor calls at once.

    Revising the top two, not just the best, keeps a runner-up with a different
    idea alive: one lucky critic score shouldn't decide the whole search.
    """
    shares = [k // len(parents) + (i < k % len(parents)) for i in range(len(parents))]
    jobs = [(p, n) for p, n in zip(parents, shares) if n]
    with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
        out = pool.map(lambda j: editor.revise(text, env_doc, j[0].source, j[0].critique, j[0].metrics, j[1]), jobs)
        return [draft for drafts in out for draft in drafts]


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
) -> str | None:
    """Return a source that compiles, asking the writer to fix errors, or None.

    Every failed attempt is logged as a "gate" event, so we can show what Claude
    got wrong and how it fixed it, even when the retry passes.
    """
    for attempt in range(GATE_RETRIES + 1):
        error = sim.gate(src)
        if error is not None:
            with lock:
                db.log_event(conn, rid, "gate", f"Candidate {idx}, attempt {attempt + 1} failed: {error}")
        if error is None:
            if attempt:
                with lock:
                    db.set_candidate(conn, cid, reward_src=src)
            return src
        if attempt == GATE_RETRIES:
            break
        src = writer.fix(src, error, env_doc)
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


def build(settings: config.Settings) -> tuple[Writer, Editor, Critic, Simulator]:
    if settings.fake_llm:
        from .agents.fake import FakeCritic, FakeEditor, FakeWriter

        writer, editor, critic = FakeWriter(), FakeEditor(), FakeCritic()
    else:
        import anthropic

        from .agents.critic import ClaudeCritic
        from .agents.editor import ClaudeEditor
        from .agents.writer import ClaudeWriter

        client = anthropic.Anthropic()
        writer = ClaudeWriter(client, settings.model)
        editor = ClaudeEditor(client, settings.model)  # swap for DroidEditor once it works
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


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    config.load_dotenv()
    settings = config.load()
    conn = db.connect(settings.db_path)
    for rid in db.fail_interrupted(conn):
        log.warning("request %s was cut off by a worker restart; marked failed", rid)
    writer, editor, critic, sim = build(settings)
    ranker = build_ranker(settings)
    log.info("worker up (fake_sim=%s, fake_llm=%s)", settings.fake_sim, settings.fake_llm)

    while True:
        req = db.claim_next_request(conn)
        if req is None:
            time.sleep(2)
            continue
        log.info("running %s: %s", req["id"], req["text"])
        try:
            best = run_request(conn, settings, req, writer, editor, critic, sim, ranker)
            db.set_request(conn, req["id"], status="done")
            db.log_event(conn, req["id"], "done", f"Finished, best score {best.score if best else 'n/a'}")
        except Exception as exc:
            log.exception("request %s failed", req["id"])
            db.set_request(conn, req["id"], status="failed")
            db.log_event(conn, req["id"], "error", f"{type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
