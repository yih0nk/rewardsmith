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
from .agents import checker as check_spec
from .agents.base import Checker, Critic, Editor, Writer
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
    checker: Checker | None = None,
) -> Best | None:
    rid, text = req["id"], req["text"]
    best: Best | None = None
    env_doc = sim.describe()
    spec = _success_spec(conn, req, checker)

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
            src = _pass_gate(conn, rid, idx, cid, src, writer, env_doc, sim)
            if src is not None:
                ready.append((cid, src, settings.artifacts_dir / rid / f"r{n}c{idx}"))
        if ready:
            db.log_event(conn, rid, "train", f"Training {len(ready)} candidate(s)")
        for cid, _, _ in ready:  # so the board can tell "waiting" from "training"
            db.set_candidate(conn, cid, status="training")
        results = sim.run_many([(src, out_dir) for _, src, out_dir in ready]) if ready else []

        round_best: Best | None = None
        for (cid, src, _), result in zip(ready, results):
            if isinstance(result, str):  # a crashed run shouldn't stop the round
                db.set_candidate(conn, cid, status="failed", error=result)
                continue
            video = _relative(result.video_path, settings.artifacts_dir)
            check = check_spec.evaluate(spec, result.metrics) if spec else None
            db.set_candidate(conn, cid, status="trained", metrics=result.metrics, video_path=video, check_result=check)

            review = critic.review(text, result, check)
            db.set_candidate(conn, cid, status="scored", score=review.score, critique=review.critique)

            cand = Best(cid, src, review.score, review.critique, result.metrics, check["pass"] if check else True)
            if round_best is None or cand.rank() > round_best.rank():
                round_best = cand

        db.finish_round(conn, round_id, round_best.candidate_id if round_best else None)
        if round_best and (best is None or round_best.rank() > best.rank()):
            best = round_best
            db.set_request(conn, rid, best_score=best.score)
        db.log_event(conn, rid, "round", f"Round {n} done, best score {best.score if best else 'n/a'}")

        if best and n >= settings.min_rounds and _succeeded(best, settings):
            break

    db.set_request(conn, rid, succeeded=int(best is not None and _succeeded(best, settings)))
    return best


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
    db.set_success_spec(conn, rid, spec)
    db.log_event(conn, rid, "check", "Success check: " + ", ".join(f"{c['metric']} {c['op']} {c['value']:g}" for c in spec))
    return spec


def _pass_gate(
    conn: sqlite3.Connection,
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
            db.log_event(conn, rid, "gate", f"Candidate {idx}, attempt {attempt + 1} failed: {error}")
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
    checker = build_checker(settings)
    log.info("worker up (fake_sim=%s, fake_llm=%s)", settings.fake_sim, settings.fake_llm)

    while True:
        req = db.claim_next_request(conn)
        if req is None:
            time.sleep(2)
            continue
        log.info("running %s: %s", req["id"], req["text"])
        try:
            best = run_request(conn, settings, req, writer, editor, critic, sim, checker)
            db.set_request(conn, req["id"], status="done")
            db.log_event(conn, req["id"], "done", f"Finished, best score {best.score if best else 'n/a'}")
        except Exception as exc:
            log.exception("request %s failed", req["id"])
            db.set_request(conn, req["id"], status="failed")
            db.log_event(conn, req["id"], "error", f"{type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
