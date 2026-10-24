# Frontend dashboard

The dashboard uses the real FastAPI backend by default. Start the API and worker
from the repository root in separate terminals:

```powershell
.venv/Scripts/python.exe -m rewardsmith.server
.venv/Scripts/python.exe -m rewardsmith.orchestrator
```

Install backend dependencies first using the root README. The default backend
configuration uses a fake simulator and fake LLM, so no GPU or API key is needed.
The fake simulator returns metrics but does not generate video.

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

Select a round and then a candidate to inspect its results. Candidate indices
from the worker are zero-based; the UI displays them starting at 1. All numeric
metrics are displayed using their returned names; `term_*` metrics appear in a
separate reward breakdown. Rounds can be empty while the writer is running.
Failed gates and training errors remain inspectable. No fixed number of rounds
or candidates is assumed by the UI.

`#<request id>` links directly to an experiment. `#hall` is reserved for the
upcoming hall of fame and shows a placeholder, without requesting an experiment
named `hall`.

## Standalone mock demo

To run without a backend, create `web/.env.local`:

```env
VITE_MOCK_API=1
```

Restart Vite after changing this setting. The mock API starts with a completed
three-round experiment. New experiments advance through five stages per round,
finishing after about 33 seconds. Mock scores and critiques are fixtures for all
prompts. The animated quadruped is an illustrative SVG, not training footage.
Mock runs reset on reload. Delete the setting or set it to `0` to return to the
backend. Production builds use the same setting at build time.

## Build

```sh
npm run build
npm test
```

This checks TypeScript and produces `dist/`. The FastAPI server serves this
bundle when it exists at startup; restart the server after the first build.

The frontend tests cover the API contract, empty rounds, candidate failures,
multiple candidates, best-round selection, metrics, reward terms, and video paths.
