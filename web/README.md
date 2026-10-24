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

The board reads top to bottom for a first-time viewer:

1. The prompt box, with click-to-fill examples and a one-line "how it works".
2. The selected run: what it is doing now in plain words ("Training 4 robots in
   parallel"), a timer for the current step and for the whole run, and a strip of
   the seven pipeline steps (set up, write rewards, test code, compile, train,
   record video, judge). Timers tick on the client clock from each stage's start
   time on the server (round, candidate and event timestamps plus each training
   process's own clock), corrected by the server's `Date` header. When the API has
   no start time for a stage, the timer counts from when the page first saw it.
   The success check is written out as a sentence.
3. The best result so far: its video, critic score, the success check clause by
   clause and Claude's critique. A run where every candidate failed shows what
   happened and what to try instead.
4. "Explore the run": collapsed rows for all candidates and the round timeline,
   one candidate in depth (learning curve, reward breakdown, measurements with
   units), the reward code, the diff, every round's winner side by side and the
   activity log.
5. All experiments.

Candidates are named "Round 2 · Candidate 3" everywhere; worker indices are
zero-based and the UI counts from 1. Terms like critic score, code check (the
gate) and revising the best rewards (carry) have hover and focus explanations.
No fixed number of rounds or candidates is assumed by the UI.

`#<request id>` links directly to an experiment. `#hall` is the hall of fame: candidates
someone flagged as reward hacking from the evaluation panel. The footer has a QR code
to the board for judges.

The board is styled as a terminal: a black page in green IBM Plex Mono (bound to the
`TerminalPlex` alias), laid out as numbered sections with the content in an output column.
Live training progress is drawn as a text bar from the real fraction, still exposed as an
ARIA progress bar. The only motion is the cursor after the wordmark, which stops under
reduced motion.

Styles live in `src/styles/`, one file per area, with the design tokens in `base.css`.

## Build

```sh
npm run build
npm test
```

This checks TypeScript and produces `dist/`. The FastAPI server serves this
bundle when it exists at startup; restart the server after the first build.

The frontend tests cover the API contract, empty rounds, candidate failures,
multiple candidates, best-round selection, metrics, reward terms, video paths,
colored diffs, flagging, the round gallery, hall-of-fame entries, every stage
label, the timers and the success check sentence.
