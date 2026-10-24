# Role C catchup: API, store and board

Role C owns `rewardsmith/server.py`, `rewardsmith/db.py`, `web/` and `tests/test_server.py`.
Corey Dai builds the dashboard design in `web/` (branch `codex/frontend-dashboard-draft`).
This file says what role C has changed and what other roles need to know. It is updated with every role C branch.

## Status

| Work | State | Branch |
|---|---|---|
| Split the board into components, reserve `#hall`, fix intermittent API 500s | Pushed; Corey's dashboard is built on it | `role-c/1-board-structure` |
| Hall of fame (API and page), restart recovery, fake reward terms, colored diffs, round gallery, QR code, startup fixes | Ready for PR | `role-c/3-hall-of-fame` |
| Video playback in Safari (HTTP range requests) | Not started | |

`role-c/3-hall-of-fame` is built on Corey's `codex/frontend-dashboard-draft` with the latest `main` merged in.
Merge Corey's branch first; then this PR shows only role C's changes.

## Contracts

What other roles read or write.

- **Reward terms:** the board draws one bar per metric whose key starts with `term_`. `PlaygroundSimulator.run()` already writes them; the fake simulator now does too.
- **Hall of fame:** `candidates` has two new columns, `flagged_at` (REAL, NULL = not flagged) and `flag_note` (TEXT). Older db files gain them when `db.connect()` opens them.
- **New endpoints:** `POST /api/candidates/{id}/flag` with `{"note": "..."}` (1 to 200 characters), `DELETE /api/candidates/{id}/flag`, `GET /api/hall-of-fame` (flagged candidates with `request_id`, `request_text`, `round_n`, newest flag first).
- **Worker startup:** `db.fail_interrupted()` marks requests left `running` as `failed`, with the event "The worker restarted mid-run…". This assumes **one worker**: starting a second worker would fail the first one's run.
- **`.env`:** the server module and the worker's `main()` call `config.load_dotenv()`, which reads `.env` from the folder the process starts in. Variables already exported win, and empty values are skipped. `config.load()` itself never reads `.env`, so tests stay isolated from a developer's `.env`.
- **URL routes:** `#<request id>` opens a request; `#hall` is the hall of fame.
- **npm:** one new dependency, `qrcode.react` 4.2.0.

## Files outside role C

| File | Owner | Change |
|---|---|---|
| `rewardsmith/orchestrator.py` | B | Lines at the top of `main()` that load `.env` and call `db.fail_interrupted()`. `run_request` is untouched. |
| `rewardsmith/sim/fake.py` | A | `run()` adds `**_fake_terms(source)` to its metrics; two new helpers at the end of the file. |
| `rewardsmith/config.py` | Shared | New `load_dotenv()`. |
| `tests/test_loop.py` | Shared | One new test at the end. |
| `.env.example`, `README.md` | Shared | Documents `.env` loading and `REWARDSMITH_SUCCESS_SCORE=11` for demos. |

## Fixes

- **Intermittent 500s from the API.** FastAPI used a request's SQLite connection from two threads. `db.connect()` passes `check_same_thread=False`; each connection still serves one request at a time.
- **Whitespace-only behaviors were accepted.** Five spaces passed the 3-character minimum and queued a request with empty text. Request text is now trimmed before validation.
- **`python -m rewardsmith.server` exited without starting.** `server.py` had no `__main__` block. web/README now uses the same commands as the root README.
- **The logo couldn't reach the home view.** The request poll re-selected the newest run every 2 s. The board now auto-selects only on first load.
- **Runs could stop after round 1.** Not a code change: the worker stops at `REWARDSMITH_SUCCESS_SCORE` (default 8). Set it to 11 for demos so every round runs.

## Merge notes

- `rewardsmith/db.py`: the `candidates` block in `SCHEMA` gained two columns. If you also add columns there, keep both sides, and add yours to `ADDED_COLUMNS` so existing db files get them.
- `rewardsmith/sim/fake.py`: if you changed the `metrics={...}` dict in `run()`, keep both sides' keys.
- `tests/test_loop.py`, `tests/test_server.py`: new tests are appended at the end; keep both sides.
- `web/package-lock.json`: take `main`'s version, run `npm install` in `web/`, and commit the result.
- `web/src/components/RequestDetail.tsx`: the reward breakdown moved into `RewardTerms.tsx`, and diffs render through `DiffView.tsx`.

## Needs from others

| What | Who | State |
|---|---|---|
| Keep Droid diffs in unified diff format (the board colors `+` and `-` lines) | B | Open |
| Add the metrics the benchmark checks read (`abs_forward_velocity`, `base_drift_m`, `height_ratio`, `airborne_all`, `hold_s`, `contact_fr`, `pitch_deg`, `fall_rate`, `abs_yaw_rate`) | A | Open |

## How to check

```sh
.venv/bin/python -m pytest -q
cd web && npm run build && npm test
```

Fake mode, with no GPU or API key: run `.venv/bin/rewardsmith-server` and `.venv/bin/rewardsmith-worker`, then `cd web && npm run dev` and open http://localhost:5173.
