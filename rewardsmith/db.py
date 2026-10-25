"""SQLite store shared by the API server and the worker.

Two processes open the same file, so the db runs in WAL mode and the worker
claims requests inside an IMMEDIATE transaction.
"""

import json
import math
import sqlite3
import threading
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS requests (
    id          TEXT PRIMARY KEY,
    text        TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'queued',  -- queued | running | paused | done | failed | cancelled
    best_score  REAL,
    best_candidate_id INTEGER,
    created_at  REAL NOT NULL,
    scene        TEXT,     -- JSON scene spec; NULL until the scene writer has run
    success_spec TEXT,     -- JSON: the pass/fail check, written once before training (agents/checker.py)
    success_check_error TEXT, -- non-null when no trustworthy success spec could be written
    succeeded    INTEGER,  -- 1 if the best candidate passed both the critic and the check
    control      TEXT      -- "pause" | "cancel": for the worker to act on at its next checkpoint
);
-- One task, as the board shows it: an ordered list of turns. Each turn is a request
-- (requests.conversation_id, requests.turn). Turn 1 is the task; later turns are follow-ups
-- that refine one of the conversation's candidates. A conversation's id is its first request's.
CREATE TABLE IF NOT EXISTS conversations (
    id          TEXT PRIMARY KEY,
    title       TEXT NOT NULL,
    created_at  REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS rounds (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id         TEXT NOT NULL REFERENCES requests(id),
    n                  INTEGER NOT NULL,
    status             TEXT NOT NULL DEFAULT 'running',  -- running | done
    best_candidate_id  INTEGER,
    created_at         REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS candidates (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    round_id     INTEGER NOT NULL REFERENCES rounds(id),
    idx          INTEGER NOT NULL,
    reward_src   TEXT NOT NULL,
    diff         TEXT,
    status       TEXT NOT NULL DEFAULT 'written',  -- written | gate_failed | trained | scored | failed
    status_at    REAL,   -- when status last changed (time.time())
    error        TEXT,
    metrics      TEXT,   -- JSON
    video_path   TEXT,   -- relative to the artifacts dir
    score        REAL,
    critique     TEXT,
    created_at   REAL NOT NULL,
    flagged_at   REAL,   -- set when someone flags it as reward hacking (hall of fame)
    flag_note    TEXT,
    check_result TEXT    -- JSON: the success check on this run, {"pass": bool, "clauses": [...]}
);
CREATE TABLE IF NOT EXISTS events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id  TEXT NOT NULL,
    kind        TEXT NOT NULL,
    message     TEXT NOT NULL,
    created_at  REAL NOT NULL
);
-- What the worker is doing for a request, with server-side start/end times, so the
-- board can run a live timer. Stages can overlap (the editor revises two parents at once).
CREATE TABLE IF NOT EXISTS stages (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id  TEXT NOT NULL,
    round_n     INTEGER,  -- NULL before round 1 (scene, check)
    name        TEXT NOT NULL,  -- scene | check | writing | compiling | training | judging
    started_at  REAL NOT NULL,
    ended_at    REAL      -- NULL while it runs
);
"""


def connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    # FastAPI may open a connection in one worker thread and use it in another.
    # Each connection still serves one request at a time, so this is safe.
    conn = sqlite3.connect(path, timeout=30, isolation_level=None, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript(SCHEMA)
    _migrate(conn)
    return conn


# Columns added after a table first shipped. CREATE TABLE IF NOT EXISTS leaves an
# existing table alone, so a db file from an older version gets them here.
ADDED_COLUMNS = {
    "requests": {"scene": "TEXT", "success_spec": "TEXT", "succeeded": "INTEGER", "success_check_error": "TEXT", "best_candidate_id": "INTEGER", "control": "TEXT",
                 "coach": "TEXT",  # the live coach's model for this request, or "off"; NULL = the worker's default
                 # Conversations (see the conversations table). NULL on rows from before them until _migrate fills them.
                 "conversation_id": "TEXT", "turn": "INTEGER",
                 "task": "TEXT",  # the behavior the agents work toward: the text, plus a follow-up's notes; NULL = text
                 "mode": "TEXT",  # instant | effort (modes.py); NULL = effort, what every run was before modes
                 "model": "TEXT",  # writer/editor/critic model; NULL = the mode's default
                 "target_candidate_id": "INTEGER",  # a follow-up's candidate to refine; NULL = the best, when it starts
                 "scene_changed": "INTEGER",  # 1 when a follow-up's note changed the scene (it trains from scratch)
                 "training_steps": "INTEGER"},  # optional per-candidate budget, including continued training
    "candidates": {
        "flagged_at": "REAL",
        "flag_note": "TEXT",
        "check_result": "TEXT",
        "status_at": "REAL",
        "idea": "TEXT",  # what the writer or editor says this candidate tries
        "reward_hacking": "INTEGER",  # 1 when the critic suspects reward hacking
    },
    # Which round and candidate (0-based idx) an event is about, when it is about one (coach events).
    "events": {"round_n": "INTEGER", "candidate": "INTEGER"},
}
JSON_COLUMNS = ("scene", "success_spec", "check_result")  # metrics is decoded in _row


def _migrate(conn: sqlite3.Connection) -> None:
    for table, columns in ADDED_COLUMNS.items():
        have = {r["name"] for r in conn.execute(f"PRAGMA table_info({table})")}
        for name, kind in columns.items():
            if name in have:
                continue
            try:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {kind}")
            except sqlite3.OperationalError as exc:  # the server and worker raced to add it
                if "duplicate column" not in str(exc):
                    raise
    # Requests from before conversations: each becomes a one-turn conversation with its own id.
    # Idempotent, so the server and the worker can both run it.
    conn.execute(
        """INSERT OR IGNORE INTO conversations (id, title, created_at)
           SELECT id, text, created_at FROM requests WHERE conversation_id IS NULL"""
    )
    conn.execute("UPDATE requests SET conversation_id = id, turn = 1 WHERE conversation_id IS NULL")


def finite_or_none(value: Any) -> Any:
    """Replace NaN and infinities with None, inside dicts and lists too.

    json.dumps writes them as NaN/Infinity, which is not JSON: strict parsers
    (JSON.parse, jq, SQLite's json functions, starlette's JSONResponse) reject the row.
    """
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: finite_or_none(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_or_none(v) for v in value]
    return value


def _row(row: sqlite3.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    out = _decode(dict(row))
    if out.get("metrics"):
        # Rows saved before set_candidate cleaned metrics can hold NaN: read it as null.
        out["metrics"] = json.loads(out["metrics"], parse_constant=lambda _: None)
    return out


def _decode(out: dict[str, Any]) -> dict[str, Any]:
    for key in JSON_COLUMNS:
        if out.get(key):
            out[key] = json.loads(out[key])
    return out


def create_request(
    conn: sqlite3.Connection, text: str, scene: dict[str, Any] | None = None, coach: str | None = None,
    mode: str | None = None, model: str | None = None, training_steps: int | None = None,
) -> str:
    """A new conversation whose first turn is this request; its id is the request's.

    scene: a validated scene spec to use as is; None lets the scene writer decide.
    coach: the live coach's model, or "off"; None leaves it to the worker's setting.
    mode, model: see modes.py; None = the defaults."""
    rid = uuid.uuid4().hex[:8]
    now = time.time()
    conn.execute("INSERT INTO conversations (id, title, created_at) VALUES (?, ?, ?)", (rid, text, now))
    conn.execute(
        """INSERT INTO requests (id, text, created_at, scene, coach, conversation_id, turn, mode, model, training_steps)
           VALUES (?, ?, ?, ?, ?, ?, 1, ?, ?, ?)""",
        (rid, text, now, json.dumps(scene) if scene is not None else None, coach, rid, mode, model, training_steps),
    )
    return rid


class NotInConversation(ValueError):
    """The conversation doesn't exist, or the candidate isn't one of its own."""


def add_turn(
    conn: sqlite3.Connection, conversation_id: str, text: str, target_candidate_id: int | None = None,
    coach: str | None = None, mode: str | None = None, model: str | None = None, training_steps: int | None = None,
) -> str:
    """A follow-up: a new request at the end of the conversation. target_candidate_id is the
    candidate to refine; None = the conversation's best when the worker starts the turn."""
    conn.execute("BEGIN IMMEDIATE")  # two follow-ups sent at once must not get the same turn number
    try:
        last = conn.execute(
            "SELECT MAX(turn) AS turn FROM requests WHERE conversation_id = ?", (conversation_id,)
        ).fetchone()["turn"]
        if last is None:
            raise NotInConversation(f"No conversation {conversation_id}")
        if target_candidate_id is not None and conversation_of_candidate(conn, target_candidate_id) != conversation_id:
            raise NotInConversation(f"Candidate {target_candidate_id} is not in conversation {conversation_id}")
        rid = uuid.uuid4().hex[:8]
        conn.execute(
            """INSERT INTO requests (id, text, created_at, coach, conversation_id, turn, mode, model, target_candidate_id, training_steps)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (rid, text, time.time(), coach, conversation_id, last + 1, mode, model, target_candidate_id, training_steps),
        )
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise
    return rid


def conversation_of_candidate(conn: sqlite3.Connection, cid: int) -> str | None:
    row = conn.execute(
        """SELECT q.conversation_id FROM candidates c JOIN rounds r ON r.id = c.round_id
           JOIN requests q ON q.id = r.request_id WHERE c.id = ?""",
        (cid,),
    ).fetchone()
    return row["conversation_id"] if row else None


def list_conversations(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    """Every conversation, newest first, with its turns (request summaries) in order."""
    turns: dict[str, list[dict[str, Any]]] = {}
    for req in sorted(list_requests(conn), key=lambda r: r["turn"] or 1):
        turns.setdefault(req["conversation_id"], []).append(req)
    out = []
    for row in conn.execute("SELECT * FROM conversations ORDER BY created_at DESC").fetchall():
        conv = dict(row)
        conv["turns"] = turns.get(conv["id"], [])
        scores = [t["best_score"] for t in conv["turns"] if t["best_score"] is not None]
        conv["best_score"] = max(scores) if scores else None
        conv["status"] = conv["turns"][-1]["status"] if conv["turns"] else None
        out.append(conv)
    return out


def scored_candidates(conn: sqlite3.Connection, conversation_id: str) -> list[dict[str, Any]]:
    """Every scored candidate in a conversation, with request_id, turn and round_n."""
    rows = conn.execute(
        """SELECT c.*, r.n AS round_n, q.id AS request_id, q.turn AS turn
           FROM candidates c JOIN rounds r ON r.id = c.round_id JOIN requests q ON q.id = r.request_id
           WHERE q.conversation_id = ? AND c.status = 'scored'
           ORDER BY q.turn, r.n, c.idx""",
        (conversation_id,),
    ).fetchall()
    return [_row(r) for r in rows]


def get_candidate(conn: sqlite3.Connection, cid: int) -> dict[str, Any] | None:
    """One candidate with its request_id, turn and round_n."""
    row = conn.execute(
        """SELECT c.*, r.n AS round_n, q.id AS request_id, q.turn AS turn
           FROM candidates c JOIN rounds r ON r.id = c.round_id JOIN requests q ON q.id = r.request_id
           WHERE c.id = ?""",
        (cid,),
    ).fetchone()
    return _row(row)


def coach_notes(conn: sqlite3.Connection, rid: str, round_n: int, idx: int) -> list[str]:
    """The live coach's notes on one candidate, oldest first."""
    rows = conn.execute(
        "SELECT message FROM events WHERE request_id = ? AND kind = 'coach' AND round_n = ? AND candidate = ? ORDER BY id",
        (rid, round_n, idx),
    ).fetchall()
    return [r["message"] for r in rows]


def list_requests(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = conn.execute("SELECT * FROM requests ORDER BY created_at DESC").fetchall()
    open_stages: dict[str, list[dict[str, Any]]] = {}
    for st in conn.execute("SELECT * FROM stages WHERE ended_at IS NULL ORDER BY id").fetchall():
        open_stages.setdefault(st["request_id"], []).append(dict(st))
    out = [_row(r) for r in rows]
    for req in out:
        req["stage"] = current_stage(open_stages.get(req["id"], []))
    return out


def get_request(conn: sqlite3.Connection, rid: str) -> dict[str, Any] | None:
    req = _row(conn.execute("SELECT * FROM requests WHERE id = ?", (rid,)).fetchone())
    if req is None:
        return None
    rounds = [_row(r) for r in conn.execute(
        "SELECT * FROM rounds WHERE request_id = ? ORDER BY n", (rid,)
    ).fetchall()]
    for rnd in rounds:
        rnd["candidates"] = [_row(c) for c in conn.execute(
            "SELECT * FROM candidates WHERE round_id = ? ORDER BY idx", (rnd["id"],)
        ).fetchall()]
    req["rounds"] = rounds
    req["events"] = [_row(e) for e in conn.execute(
        "SELECT * FROM events WHERE request_id = ? ORDER BY id", (rid,)
    ).fetchall()]
    req["stages"] = list_stages(conn, rid)
    req["stage"] = current_stage(req["stages"])
    return req


def claim_next_request(conn: sqlite3.Connection) -> dict[str, Any] | None:
    conn.execute("BEGIN IMMEDIATE")
    try:
        row = conn.execute(
            "SELECT * FROM requests WHERE status = 'queued' ORDER BY created_at LIMIT 1"
        ).fetchone()
        if row is not None:
            conn.execute("UPDATE requests SET status = 'running' WHERE id = ?", (row["id"],))
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise
    return _row(row)


def set_request(conn: sqlite3.Connection, rid: str, **fields: Any) -> None:
    if fields.get("scene") is not None:
        fields["scene"] = json.dumps(fields["scene"])
    _update(conn, "requests", "id", rid, fields)


def set_success_spec(conn: sqlite3.Connection, rid: str, spec: list[dict[str, Any]]) -> bool:
    """Store the request's success check. Only once: False if it already has one,
    so the target can't move after results come in."""
    cur = conn.execute(
        "UPDATE requests SET success_spec = ? WHERE id = ? AND success_spec IS NULL", (json.dumps(spec), rid)
    )
    return cur.rowcount > 0


def add_round(conn: sqlite3.Connection, rid: str, n: int) -> int:
    cur = conn.execute(
        "INSERT INTO rounds (request_id, n, created_at) VALUES (?, ?, ?)", (rid, n, time.time())
    )
    return int(cur.lastrowid)


def finish_round(conn: sqlite3.Connection, round_id: int, best_candidate_id: int | None) -> None:
    _update(conn, "rounds", "id", round_id, {"status": "done", "best_candidate_id": best_candidate_id})


def add_candidate(
    conn: sqlite3.Connection,
    round_id: int,
    idx: int,
    reward_src: str,
    diff: str | None = None,
    idea: str | None = None,
) -> int:
    now = time.time()
    cur = conn.execute(
        "INSERT INTO candidates (round_id, idx, reward_src, diff, idea, created_at, status_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (round_id, idx, reward_src, diff, idea, now, now),
    )
    return int(cur.lastrowid)


def set_candidate(conn: sqlite3.Connection, cid: int, **fields: Any) -> None:
    if "status" in fields:
        fields.setdefault("status_at", time.time())
    if fields.get("check_result") is not None:
        fields["check_result"] = json.dumps(fields["check_result"])
    if "metrics" in fields and fields["metrics"] is not None:
        fields["metrics"] = json.dumps(finite_or_none(fields["metrics"]), allow_nan=False)
    _update(conn, "candidates", "id", cid, fields)


def log_event(
    conn: sqlite3.Connection, rid: str, kind: str, message: str,
    round_n: int | None = None, candidate: int | None = None,
) -> None:
    """candidate is the 0-based idx within round_n, for events about one candidate."""
    conn.execute(
        "INSERT INTO events (request_id, kind, message, created_at, round_n, candidate) VALUES (?, ?, ?, ?, ?, ?)",
        (rid, kind, message, time.time(), round_n, candidate),
    )


_stage_lock = threading.Lock()  # stages open and close from the orchestrator's thread pools


def start_stage(conn: sqlite3.Connection, rid: str, name: str, round_n: int | None = None) -> int:
    with _stage_lock:
        cur = conn.execute(
            "INSERT INTO stages (request_id, round_n, name, started_at) VALUES (?, ?, ?, ?)",
            (rid, round_n, name, time.time()),
        )
        return int(cur.lastrowid)


def end_stage(conn: sqlite3.Connection, stage_id: int) -> None:
    with _stage_lock:
        conn.execute("UPDATE stages SET ended_at = ? WHERE id = ? AND ended_at IS NULL", (time.time(), stage_id))


@contextmanager
def stage(conn: sqlite3.Connection, rid: str, name: str, round_n: int | None = None) -> Iterator[int]:
    """Record a stage around a block: started_at now, ended_at when the block exits (even on error)."""
    stage_id = start_stage(conn, rid, name, round_n)
    try:
        yield stage_id
    finally:
        end_stage(conn, stage_id)


def list_stages(conn: sqlite3.Connection, rid: str) -> list[dict[str, Any]]:
    return [dict(r) for r in conn.execute("SELECT * FROM stages WHERE request_id = ? ORDER BY id", (rid,)).fetchall()]


def current_stage(stages: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The most recently started stage that is still open, or None."""
    open_ = [s for s in stages if s["ended_at"] is None]
    return open_[-1] if open_ else None


def fail_interrupted(conn: sqlite3.Connection) -> list[str]:
    """Fail requests a stopped worker left running. Call once at worker startup.

    Assumes one worker: a second worker starting up would fail the first one's run.
    """
    ids = [r["id"] for r in conn.execute("SELECT id FROM requests WHERE status = 'running'").fetchall()]
    for rid in ids:
        fail_request(conn, rid, "The worker restarted mid-run. Submit the behavior again to retry.")
    return ids


# Candidate statuses that mean "still in progress". A failed request must not leave any
# behind, or the board keeps showing them as training or being judged.
UNFINISHED = ("written", "training", "trained")


def fail_request(conn: sqlite3.Connection, rid: str, reason: str) -> None:
    """Mark a request failed, with its unfinished rounds closed and unfinished candidates failed."""
    set_request(conn, rid, status="failed", control=None)
    _close_unfinished(conn, rid, "failed", "Stopped: the run failed before this candidate finished.")
    log_event(conn, rid, "error", reason)


def _close_unfinished(conn: sqlite3.Connection, rid: str, status: str, error: str) -> None:
    conn.execute(
        f"""UPDATE candidates SET status = ?, status_at = ?, error = COALESCE(error, ?)
            WHERE status IN ({",".join("?" * len(UNFINISHED))})
              AND round_id IN (SELECT id FROM rounds WHERE request_id = ?)""",
        (status, time.time(), error, *UNFINISHED, rid),
    )
    conn.execute("UPDATE rounds SET status = 'done' WHERE request_id = ? AND status = 'running'", (rid,))
    conn.execute("UPDATE stages SET ended_at = ? WHERE request_id = ? AND ended_at IS NULL", (time.time(), rid))


# Pause, resume and cancel. A queued or paused request changes at once; a running one gets a
# `control` the worker acts on at its next checkpoint (pause between rounds, cancel sooner).
# Each change is one conditional UPDATE, so it can't race the worker claiming the request.
# They return the new state, or None when the request is not in a state that allows it.

def pause_request(conn: sqlite3.Connection, rid: str) -> str | None:
    if _change(conn, "UPDATE requests SET status = 'paused' WHERE id = ? AND status = 'queued'", rid):
        log_event(conn, rid, "control", "Paused before it started")
        return "paused"
    if _change(conn, "UPDATE requests SET control = 'pause' WHERE id = ? AND status = 'running' AND control IS NULL", rid):
        log_event(conn, rid, "control", "Pausing after the current round")
        return "pausing"
    return None


def resume_request(conn: sqlite3.Connection, rid: str) -> str | None:
    if _change(conn, "UPDATE requests SET status = 'queued', control = NULL WHERE id = ? AND status = 'paused'", rid):
        log_event(conn, rid, "control", "Resumed: waiting for the worker")
        return "queued"
    if _change(conn, "UPDATE requests SET control = NULL WHERE id = ? AND status = 'running' AND control = 'pause'", rid):
        log_event(conn, rid, "control", "Pause withdrawn")
        return "running"
    return None


def cancel_request(conn: sqlite3.Connection, rid: str) -> str | None:
    if _change(conn, "UPDATE requests SET status = 'cancelled', control = NULL WHERE id = ? AND status IN ('queued', 'paused')", rid):
        _close_unfinished(conn, rid, "stopped", CANCELLED)
        log_event(conn, rid, "control", "Cancelled")
        return "cancelled"
    if _change(conn, "UPDATE requests SET control = 'cancel' WHERE id = ? AND status = 'running' AND control IS NOT 'cancel'", rid):
        log_event(conn, rid, "control", "Cancelling at the next checkpoint")
        return "cancelling"
    return None


def finish_paused(conn: sqlite3.Connection, rid: str, rounds_done: int) -> None:
    """The worker reached a round boundary with a pause pending."""
    set_request(conn, rid, status="paused", control=None)
    log_event(conn, rid, "control", f"Paused after round {rounds_done}. Resume to continue.")


def finish_cancelled(conn: sqlite3.Connection, rid: str) -> None:
    """The worker reached a checkpoint with a cancel pending."""
    set_request(conn, rid, status="cancelled", control=None)
    _close_unfinished(conn, rid, "stopped", CANCELLED)
    log_event(conn, rid, "control", "Cancelled")


def control_of(conn: sqlite3.Connection, rid: str) -> str | None:
    row = conn.execute("SELECT control FROM requests WHERE id = ?", (rid,)).fetchone()
    return row["control"] if row else None


CANCELLED = "Cancelled before this candidate finished."


def _change(conn: sqlite3.Connection, sql: str, rid: str) -> bool:
    return conn.execute(sql, (rid,)).rowcount > 0


def flag_candidate(conn: sqlite3.Connection, cid: int, note: str) -> bool:
    """Add a candidate to the hall of fame. False if there is no such candidate."""
    cur = conn.execute(
        "UPDATE candidates SET flagged_at = ?, flag_note = ? WHERE id = ?", (time.time(), note, cid)
    )
    return cur.rowcount > 0


def unflag_candidate(conn: sqlite3.Connection, cid: int) -> bool:
    cur = conn.execute("UPDATE candidates SET flagged_at = NULL, flag_note = NULL WHERE id = ?", (cid,))
    return cur.rowcount > 0


def list_flagged(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    """Flagged candidates with their request and round, newest flag first."""
    rows = conn.execute(
        """SELECT c.*, r.n AS round_n, q.id AS request_id, q.text AS request_text
           FROM candidates c
           JOIN rounds r ON r.id = c.round_id
           JOIN requests q ON q.id = r.request_id
           WHERE c.flagged_at IS NOT NULL
           ORDER BY c.flagged_at DESC"""
    ).fetchall()
    return [_row(r) for r in rows]


def _update(conn: sqlite3.Connection, table: str, key: str, value: Any, fields: dict[str, Any]) -> None:
    if not fields:
        return
    cols = ", ".join(f"{k} = ?" for k in fields)
    conn.execute(f"UPDATE {table} SET {cols} WHERE {key} = ?", (*fields.values(), value))
