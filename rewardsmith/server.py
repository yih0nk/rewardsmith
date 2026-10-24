"""HTTP API and static hosting for the board.

On Runpod, expose port 8000; the pod's proxy URL is the board link judges open.
"""

from collections.abc import Iterator
from pathlib import Path
import sqlite3

from fastapi import Depends, FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from . import config, db

settings = config.load()
settings.artifacts_dir.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Rewardsmith")


def get_conn() -> Iterator[sqlite3.Connection]:
    conn = db.connect(settings.db_path)
    try:
        yield conn
    finally:
        conn.close()


class NewRequest(BaseModel):
    text: str = Field(min_length=3, max_length=300)


@app.post("/api/requests", status_code=201)
def create_request(body: NewRequest, conn: sqlite3.Connection = Depends(get_conn)) -> dict:
    rid = db.create_request(conn, body.text.strip())
    db.log_event(conn, rid, "queued", "Request queued")
    return {"id": rid}


@app.get("/api/requests")
def list_requests(conn: sqlite3.Connection = Depends(get_conn)) -> list[dict]:
    return db.list_requests(conn)


@app.get("/api/requests/{rid}")
def get_request(rid: str, conn: sqlite3.Connection = Depends(get_conn)) -> dict:
    req = db.get_request(conn, rid)
    if req is None:
        raise HTTPException(status_code=404, detail="No request with that id")
    return req


class Flag(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    note: str = Field(min_length=1, max_length=200)


@app.post("/api/candidates/{cid}/flag")
def flag_candidate(cid: int, body: Flag, conn: sqlite3.Connection = Depends(get_conn)) -> dict:
    if not db.flag_candidate(conn, cid, body.note):
        raise HTTPException(status_code=404, detail="No candidate with that id")
    return {"ok": True}


@app.delete("/api/candidates/{cid}/flag")
def unflag_candidate(cid: int, conn: sqlite3.Connection = Depends(get_conn)) -> dict:
    if not db.unflag_candidate(conn, cid):
        raise HTTPException(status_code=404, detail="No candidate with that id")
    return {"ok": True}


@app.get("/api/hall-of-fame")
def hall_of_fame(conn: sqlite3.Connection = Depends(get_conn)) -> list[dict]:
    return db.list_flagged(conn)


@app.get("/api/health")
def health() -> dict:
    return {"ok": True, "fake_sim": settings.fake_sim, "fake_llm": settings.fake_llm}


app.mount("/artifacts", StaticFiles(directory=settings.artifacts_dir), name="artifacts")

_web_dist = Path(__file__).resolve().parent.parent / "web" / "dist"
if _web_dist.exists():
    app.mount("/", StaticFiles(directory=_web_dist, html=True), name="web")


def main() -> None:
    import uvicorn

    uvicorn.run("rewardsmith.server:app", host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
