# Frontend dashboard

The dashboard uses the FastAPI backend. Start the API and worker
from the repository root in separate terminals:

```sh
.venv/bin/rewardsmith-server
.venv/bin/rewardsmith-worker
```

On Windows the same commands are `.venv\Scripts\rewardsmith-server.exe` and
`.venv\Scripts\rewardsmith-worker.exe`. `python -m rewardsmith.server` and
`python -m rewardsmith.orchestrator` also work.

Install backend dependencies first using the root README. The default backend
configuration uses a fake simulator and fake LLM, so no GPU or API key is needed.
The fake simulator returns metrics, including `term_*` reward terms, but does not
generate video.

Then start the frontend:

```sh
cd web
npm ci
npm run dev
```

Open the URL Vite prints (normally http://localhost:5173). Vite proxies `/api` and
`/artifacts` to the backend on port 8000. The mode badge comes from `/api/health`.
Requests are persisted by the backend; the worker consumes queued requests.

## Backend contract

- `GET /api/health`: API availability, `fake_sim`, and `fake_llm`.
- `GET /api/requests`: recent experiments, statuses, and best scores.
- `POST /api/requests` with `{ "text": "walk backwards" }`: returns `{ "id": "..." }`.
- `GET /api/requests/{id}`: rounds, all candidates, metrics, critiques, and events.
- `/artifacts/{video_path}`: recorded videos when the simulator provides them.
- `POST /api/candidates/{id}/flag` with `{ "note": "..." }` (1 to 200 characters) and
  `DELETE /api/candidates/{id}/flag`: add a candidate to, or remove it from, the hall of fame.
  Candidates carry `flagged_at` and `flag_note`.
- `GET /api/hall-of-fame`: flagged candidates, newest flag first, each with
  `request_id`, `request_text` and `round_n`.

Select a round and then a candidate to inspect its results. Candidate indices
from the worker are zero-based; the UI displays them starting at 1. All numeric
metrics are displayed using their returned names; `term_*` metrics appear in a
separate reward breakdown. Rounds can be empty while the writer is running.
Failed gates and training errors remain inspectable. No fixed number of rounds
or candidates is assumed by the UI.

`#<request id>` links directly to an experiment. `#hall` is the hall of fame: candidates
someone flagged as reward hacking from the evaluation panel. A completed run with two or
more rollout videos also shows each round's winner side by side. The home view has a QR
code to the board for judges; it is hidden on phones.

## Build

```sh
npm run build
npm test
```

This checks TypeScript and produces `dist/`. The FastAPI server serves this
bundle when it exists at startup; restart the server after the first build.

The frontend tests cover the API contract, empty rounds, candidate failures,
multiple candidates, best-round selection, metrics, reward terms, video paths,
colored diffs, flagging, the round gallery and hall-of-fame entries.
