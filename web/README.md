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

The board works like a chat app. Each task is a conversation.

- **Left sidebar:** a New task button, then one row per task, newest first,
  with its status (queued, running, done, no result, failed) and best score.
  Below 900 px it becomes a drawer, opened from the header's Tasks button.
- **New task:** what the board does, then a composer pinned to the bottom of the
  pane, with click-to-fill examples. If the GPU is busy, it names the task that a
  new one will wait for.
- **A task:** a thread of turns. Today each request is one conversation with one
  turn: the user's message, then the run card:
  1. What the run is doing now, in plain words ("Training 4 robots in
     parallel"; a queued task says "Queued, starts when <task> finishes"). A timer
     for the current step and one for the whole run, and a strip of the seven
     pipeline steps (set up, write rewards, test code, compile, train, record
     video, judge). Timers tick on the client clock from each stage's start time on
     the server (round, candidate and event timestamps plus each training
     process's own clock), corrected by the server's `Date` header. When the API
     has no start time for a stage, the timer counts from when the page first saw
     it. The success check is written out as a sentence.
  2. The best result so far: its video, critic score, the success check clause
     by clause and Claude's critique. When the pane is 820 px or wider, it sits
     beside the status. A run where every candidate failed shows what happened
     and what to try instead.
  3. "Explore the run": collapsed rows for all candidates and the round
     timeline, one candidate in depth (learning curve, reward breakdown,
     measurements with units), the reward code, the diff, every round's winner
     side by side and the activity log.

Candidates are named "Round 2 · Candidate 3" everywhere; worker indices are
zero-based and the UI counts from 1. Terms like critic score, code check (the
gate) and revising the best rewards (carry) have hover and focus explanations.
No fixed number of rounds or candidates is assumed by the UI.

Routes live in the hash: `#<request id>` opens a task, `#new` the new-task pane
and `#hall` the hall of fame (candidates someone flagged as reward hacking). The
new-task pane has a QR code to the board for judges.

### Hooks for follow-ups

The backend has no follow-ups yet, so none of these render a control today:

- `src/thread.ts`: `conversationsFromRequests` groups requests into
  conversations (today one request per conversation). `ReplyTarget` is what a
  reply aims at: the whole run, or one candidate
  (`requestId`, `round`, `candidateId`, `idx`). `ComposerOptions` is the
  per-message mode and model.
- `RequestDetail` (one turn) takes `renderReply`. When set, it fills reply slots
  on the run, on the result shown, and on each candidate card. Runs have the DOM
  id `run-<id>`; candidates have `candidate-<id>` with `data-round` and
  `data-candidate-id`.
- `ConversationView` takes a `composer` node, rendered under the last turn. Pass
  a `Composer` once follow-ups work.
- `Composer` takes `controls`, rendered under the input. That is where the
  Instant/Effort toggle and the model picker go.

The board is styled as a terminal: a black page in green IBM Plex Mono (bound to the
`TerminalPlex` alias), laid out as numbered turns with the content in an output column.
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
label, the timers, the success check sentence, the conversation list, queue
labels and reply targets.
