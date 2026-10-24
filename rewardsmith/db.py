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
    error        TEXT,
    metrics      TEXT,   -- JSON
    video_path   TEXT,   -- relative to the artifacts dir
    score        REAL,
    critique     TEXT,
    created_at   REAL NOT NULL
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
    conn = sqlite3.connect(path, timeout=30, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript(SCHEMA)
    return conn


def _row(row: sqlite3.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    out = dict(row)
    if out.get("metrics"):
        out["metrics"] = json.loads(out["metrics"])
    return out


def create_request(conn: sqlite3.Connection, text: str) -> str:
    rid = uuid.uuid4().hex[:8]
    conn.execute(
        "INSERT INTO requests (id, text, created_at) VALUES (?, ?, ?)",
        (rid, text, time.time()),
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


def _update(conn: sqlite3.Connection, table: str, key: str, value: Any, fields: dict[str, Any]) -> None:
    if not fields:
        return
    cols = ", ".join(f"{k} = ?" for k in fields)
    conn.execute(f"UPDATE {table} SET {cols} WHERE {key} = ?", (*fields.values(), value))
