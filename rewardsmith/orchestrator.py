"""The worker: claims a request and runs write -> gate -> train -> critique rounds.

Round 1: the writer drafts K rewards from the request.
Round 2+: the editor revises the best reward so far, using the critic's notes.
"""

import logging
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import config, db
from .agents.base import Critic, Editor, Writer
from .sim.base import Simulator

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
) -> Best | None:
    rid, text = req["id"], req["text"]
    best: Best | None = None
    env_doc = sim.describe()

    for n in range(1, settings.max_rounds + 1):
        round_id = db.add_round(conn, rid, n)
        db.log_event(conn, rid, "round", f"Round {n} started")

        if best is None:
            drafts = [(src, None) for src in writer.write(text, env_doc, settings.candidates_per_round)]
        else:
            drafts = editor.revise(
                text, env_doc, best.source, best.critique, best.metrics, settings.candidates_per_round
            )

        # Gate every draft first, then train the survivors together: on the GPU,
        # sim.run_many trains them side by side, which hides most of the JIT time.
        ready: list[tuple[int, str, Path]] = []
        for idx, (src, diff) in enumerate(drafts):
            cid = db.add_candidate(conn, round_id, idx, src, diff)
            src = _pass_gate(conn, cid, src, writer, env_doc, sim)
            if src is not None:
                ready.append((cid, src, settings.artifacts_dir / rid / f"r{n}c{idx}"))
        if ready:
            db.log_event(conn, rid, "train", f"Training {len(ready)} candidate(s)")
        results = sim.run_many([(src, out_dir) for _, src, out_dir in ready]) if ready else []

        round_best: Best | None = None
        for (cid, src, _), result in zip(ready, results):
            if isinstance(result, str):  # a crashed run shouldn't stop the round
                db.set_candidate(conn, cid, status="failed", error=result)
                continue
            video = _relative(result.video_path, settings.artifacts_dir)
            db.set_candidate(conn, cid, status="trained", metrics=result.metrics, video_path=video)

            review = critic.review(text, result)
            db.set_candidate(conn, cid, status="scored", score=review.score, critique=review.critique)

            if round_best is None or review.score > round_best.score:
                round_best = Best(cid, src, review.score, review.critique, result.metrics)

        db.finish_round(conn, round_id, round_best.candidate_id if round_best else None)
        if round_best and (best is None or round_best.score > best.score):
            best = round_best
            db.set_request(conn, rid, best_score=best.score)
        db.log_event(conn, rid, "round", f"Round {n} done, best score {best.score if best else 'n/a'}")

        if best and best.score >= settings.success_score:
            break

    return best


def _pass_gate(
    conn: sqlite3.Connection, cid: int, src: str, writer: Writer, env_doc: str, sim: Simulator
) -> str | None:
    """Return a source that compiles, asking the writer to fix errors, or None."""
    for attempt in range(GATE_RETRIES + 1):
        error = sim.gate(src)
        if error is None:
            if attempt:
                db.set_candidate(conn, cid, reward_src=src)
            return src
        if attempt == GATE_RETRIES:
            break
        src = writer.fix(src, error, env_doc)
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


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    settings = config.load()
    conn = db.connect(settings.db_path)
    writer, editor, critic, sim = build(settings)
    log.info("worker up (fake_sim=%s, fake_llm=%s)", settings.fake_sim, settings.fake_llm)

    while True:
        req = db.claim_next_request(conn)
        if req is None:
            time.sleep(2)
            continue
        log.info("running %s: %s", req["id"], req["text"])
        try:
            best = run_request(conn, settings, req, writer, editor, critic, sim)
            db.set_request(conn, req["id"], status="done")
            db.log_event(conn, req["id"], "done", f"Finished, best score {best.score if best else 'n/a'}")
        except Exception as exc:
            log.exception("request %s failed", req["id"])
            db.set_request(conn, req["id"], status="failed")
            db.log_event(conn, req["id"], "error", f"{type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
