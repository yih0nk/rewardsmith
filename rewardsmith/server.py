"""HTTP API and static hosting for the board.

On Runpod, expose port 8000; the pod's proxy URL is the board link judges open.
"""

from collections.abc import Iterator
from pathlib import Path
import sqlite3
import time
from typing import Any

from fastapi import Depends, FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from . import config, db
from .sim import progress, scene

config.load_dotenv()  # here, not in main(): the console script imports this module first
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
    model_config = ConfigDict(str_strip_whitespace=True)  # "   ab" must not pass min_length

    text: str = Field(min_length=3, max_length=300)
    # Optional scene spec (see rewardsmith/sim/scene.py). Omit it to let the scene writer decide.
    scene: dict[str, Any] | None = None


@app.post("/api/requests", status_code=201)
def create_request(body: NewRequest, conn: sqlite3.Connection = Depends(get_conn)) -> dict:
    spec = None
    if body.scene is not None:
        try:
            spec = scene.parse(body.scene).to_dict()
        except scene.SceneError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
    rid = db.create_request(conn, body.text, spec)
    db.log_event(conn, rid, "queued", "Request queued")
    return {"id": rid}


@app.get("/api/requests")
def list_requests(conn: sqlite3.Connection = Depends(get_conn)) -> list[dict]:
    return db.list_requests(conn)


CONTROLS = {"pause": db.pause_request, "resume": db.resume_request, "cancel": db.cancel_request}


@app.post("/api/requests/{rid}/{action}")
def control_request(rid: str, action: str, conn: sqlite3.Connection = Depends(get_conn)) -> dict:
    """pause | resume | cancel. Returns the new state ("pausing"/"cancelling" while the
    worker finishes its current step); 409 when the request's state doesn't allow it."""
    if action not in CONTROLS:
        raise HTTPException(status_code=404, detail="Unknown action")
    req = db.get_request(conn, rid)
    if req is None:
        raise HTTPException(status_code=404, detail="No request with that id")
    state = CONTROLS[action](conn, rid)
    if state is None:
        raise HTTPException(status_code=409, detail=f"Can't {action} a {req['status']} request")
    return {"state": state}


@app.get("/api/requests/{rid}")
def get_request(rid: str, conn: sqlite3.Connection = Depends(get_conn)) -> dict:
    req = db.get_request(conn, rid)
    if req is None:
        raise HTTPException(status_code=404, detail="No request with that id")
    # The board times live stages against this clock, not the viewer's.
    req["server_time"] = time.time()
    # Live training progress, written by the training process (null until it starts).
    for rnd in req["rounds"]:
        for cand in rnd["candidates"]:
            out_dir = progress.candidate_dir(settings.artifacts_dir, rid, rnd["n"], cand["idx"])
            cand["progress"] = progress.read_progress(out_dir)
            cand["robustness"] = robustness_summary(cand.get("metrics"))
    return req


ROBUSTNESS_GROUPS = ("robots", "seeds", "pushes")


def robustness_summary(metrics: dict | None) -> dict | None:
    """{"robots": {"passed": 8, "total": 10}, ...} from the robust_* metrics; null if not scored."""
    out = {
        group: {"passed": metrics[f"robust_{group}_pass"], "total": metrics[f"robust_{group}_n"]}
        for group in ROBUSTNESS_GROUPS
        if metrics and f"robust_{group}_n" in metrics
    }
    return out or None


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
