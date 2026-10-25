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

from . import config, db, modes
from .sim import bodies, progress, scene

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
    # The live coach that watches training: a Claude model id, or "off". Omit it for the worker's default.
    coach: str | None = Field(default=None, pattern=r"^(off|claude-[a-z0-9-]+)$")
    # instant | effort (modes.py). Omit it for the default: Effort for a new task.
    mode: str | None = Field(default=None, pattern=f"^({modes.INSTANT}|{modes.EFFORT})$")
    # The writer/editor/critic model, one of modes.MODELS. Omit it for the mode's.
    model: str | None = Field(default=None, pattern="^(" + "|".join(modes.MODELS) + ")$")
    # The robot: a registered body's name (GET /api/health lists them), or "auto". Omit it for Auto:
    # Claude picks the body that fits the task. Follow-ups always run on the conversation's body.
    body: str | None = Field(default=None, max_length=40)


@app.post("/api/requests", status_code=201)
def create_request(body: NewRequest, conn: sqlite3.Connection = Depends(get_conn)) -> dict:
    """A new task: a conversation whose first turn is this request (same id)."""
    spec = None
    if body.scene is not None:
        try:
            spec = scene.parse(body.scene).to_dict()
        except scene.SceneError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
    robot = None if body.body in (None, "", "auto") else body.body
    if robot is not None and robot not in bodies.BODIES:
        raise HTTPException(status_code=422, detail=f"Unknown body {robot!r}: use auto or one of {sorted(bodies.BODIES)}")
    rid = db.create_request(conn, body.text, spec, body.coach, body.mode or modes.DEFAULT_TASK_MODE, body.model, robot)
    db.log_event(conn, rid, "queued", "Request queued")
    return {"id": rid, "conversation_id": rid}


class NewTurn(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    # The user's note: "get behind the ball first", "lift your feet higher". Empty = keep going.
    text: str = Field(default="Keep going", max_length=300)
    # The candidate to refine. Omit it for the conversation's best when the turn starts.
    target_candidate_id: int | None = None
    coach: str | None = Field(default=None, pattern=r"^(off|claude-[a-z0-9-]+)$")
    # Omit it for the default: Instant for a follow-up.
    mode: str | None = Field(default=None, pattern=f"^({modes.INSTANT}|{modes.EFFORT})$")
    model: str | None = Field(default=None, pattern="^(" + "|".join(modes.MODELS) + ")$")


@app.post("/api/conversations/{conversation_id}/turns", status_code=201)
def add_turn(conversation_id: str, body: NewTurn, conn: sqlite3.Connection = Depends(get_conn)) -> dict:
    """A follow-up: refine one of the conversation's candidates. It queues like any request."""
    try:
        rid = db.add_turn(
            conn, conversation_id, body.text or "Keep going", body.target_candidate_id, body.coach,
            body.mode or modes.DEFAULT_FOLLOWUP_MODE, body.model,
        )
    except db.NotInConversation as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    db.log_event(conn, rid, "queued", "Follow-up queued")
    return {"id": rid, "conversation_id": conversation_id}


@app.get("/api/conversations")
def list_conversations(conn: sqlite3.Connection = Depends(get_conn)) -> list[dict]:
    """Every task, newest first, each with its turns (request summaries) in order."""
    return db.list_conversations(conn)


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
    # A follow-up's candidate, so the board can name it ("Turn 1 · Round 2 · Candidate 3").
    target = db.get_candidate(conn, req["target_candidate_id"]) if req.get("target_candidate_id") else None
    req["target"] = (
        {k: target[k] for k in ("id", "request_id", "turn", "round_n", "idx", "score")} if target else None
    )
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
    return {
        "ok": True, "fake_sim": settings.fake_sim, "fake_llm": settings.fake_llm,
        "candidates_per_round": settings.candidates_per_round, "max_rounds": settings.max_rounds,
        **modes.describe(),
        # The robots a request can name, default first; only registered bodies are ever offered.
        "default_body": bodies.DEFAULT_BODY,
        "bodies": [
            {"name": b.name, "title": b.title, "kind": b.capabilities.kind, "summary": b.capabilities.summary}
            for b in sorted(bodies.BODIES.values(), key=lambda b: b.name != bodies.DEFAULT_BODY)
        ],
    }


app.mount("/artifacts", StaticFiles(directory=settings.artifacts_dir), name="artifacts")

_web_dist = Path(__file__).resolve().parent.parent / "web" / "dist"
if _web_dist.exists():
    app.mount("/", StaticFiles(directory=_web_dist, html=True), name="web")


def main() -> None:
    import uvicorn

    uvicorn.run("rewardsmith.server:app", host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
