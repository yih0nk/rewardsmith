"""SQLite store shared by the API server and the worker.

Two processes open the same file, so the db runs in WAL mode and the worker
claims requests inside an IMMEDIATE transaction.
"""

import json
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS requests (
    id          TEXT PRIMARY KEY,
    text        TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'queued',  -- queued | running | done | failed
    best_score  REAL,
    created_at  REAL NOT NULL,
    scene       TEXT   -- JSON scene spec; NULL until the scene writer has run
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
    error        TEXT,
    metrics      TEXT,   -- JSON
    video_path   TEXT,   -- relative to the artifacts dir
    score        REAL,
    critique     TEXT,
    created_at   REAL NOT NULL,
    flagged_at   REAL,   -- set when someone flags it as reward hacking (hall of fame)
    flag_note    TEXT
);
CREATE TABLE IF NOT EXISTS events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id  TEXT NOT NULL,
    kind        TEXT NOT NULL,
    message     TEXT NOT NULL,
    created_at  REAL NOT NULL
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
ADDED_COLUMNS = {"candidates": {"flagged_at": "REAL", "flag_note": "TEXT"}, "requests": {"scene": "TEXT"}}


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


def _row(row: sqlite3.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    out = dict(row)
    for key in ("metrics", "scene"):
        if out.get(key):
            out[key] = json.loads(out[key])
    return out


def create_request(conn: sqlite3.Connection, text: str, scene: dict[str, Any] | None = None) -> str:
    """scene: a validated scene spec to use as is; None lets the scene writer decide."""
    rid = uuid.uuid4().hex[:8]
    conn.execute(
        "INSERT INTO requests (id, text, created_at, scene) VALUES (?, ?, ?, ?)",
        (rid, text, time.time(), json.dumps(scene) if scene is not None else None),
    )
    return rid


def list_requests(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = conn.execute("SELECT * FROM requests ORDER BY created_at DESC").fetchall()
    return [_row(r) for r in rows]


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


def add_round(conn: sqlite3.Connection, rid: str, n: int) -> int:
    cur = conn.execute(
        "INSERT INTO rounds (request_id, n, created_at) VALUES (?, ?, ?)", (rid, n, time.time())
    )
    return int(cur.lastrowid)


def finish_round(conn: sqlite3.Connection, round_id: int, best_candidate_id: int | None) -> None:
    _update(conn, "rounds", "id", round_id, {"status": "done", "best_candidate_id": best_candidate_id})


def add_candidate(
    conn: sqlite3.Connection, round_id: int, idx: int, reward_src: str, diff: str | None = None
) -> int:
    cur = conn.execute(
        "INSERT INTO candidates (round_id, idx, reward_src, diff, created_at) VALUES (?, ?, ?, ?, ?)",
        (round_id, idx, reward_src, diff, time.time()),
    )
    return int(cur.lastrowid)


def set_candidate(conn: sqlite3.Connection, cid: int, **fields: Any) -> None:
    if "metrics" in fields and fields["metrics"] is not None:
        fields["metrics"] = json.dumps(fields["metrics"])
    _update(conn, "candidates", "id", cid, fields)


def log_event(conn: sqlite3.Connection, rid: str, kind: str, message: str) -> None:
    conn.execute(
        "INSERT INTO events (request_id, kind, message, created_at) VALUES (?, ?, ?, ?)",
        (rid, kind, message, time.time()),
    )


def fail_interrupted(conn: sqlite3.Connection) -> list[str]:
    """Fail requests a stopped worker left running. Call once at worker startup.

    Assumes one worker: a second worker starting up would fail the first one's run.
    """
    ids = [r["id"] for r in conn.execute("SELECT id FROM requests WHERE status = 'running'").fetchall()]
    for rid in ids:
        set_request(conn, rid, status="failed")
        log_event(conn, rid, "error", "The worker restarted mid-run. Submit the behavior again to retry.")
    return ids


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
