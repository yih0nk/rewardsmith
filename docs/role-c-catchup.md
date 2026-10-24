# Role C catchup: API, store and board

Role C owns `rewardsmith/server.py`, `rewardsmith/db.py`, `web/` and `tests/test_server.py`.
This file says what role C has changed and what other roles need to know. It is updated at the end of every phase.

## Status

| Phase | What | State | Branch |
|---|---|---|---|
| 1 | Split the board into components, add the `#hall` route | In PR | `role-c/1-board-structure` |
| 2 | Score chart, reward-term bars, colored diffs | Not started | `role-c/2-charts` |
| 3 | Hall of fame: flag a candidate, list flagged runs | Not started | `role-c/3-hall-of-fame` |
| 4 | Phone layout and a QR code to the board | Not started | `role-c/4-phone-qr` |
| 5 | Video playback in Safari (HTTP range requests) | Not started | `role-c/5-video` |

## Contracts

What other roles read or write. Nothing here changes the API or the database yet.

- **URL routes:** `#<request id>` opens a request (unchanged). `#hall` is reserved for the hall of fame, so don't use `hall` as a request id.
- **Reward terms (phase 2):** the board will draw one bar per metric whose key starts with `term_`, which is what `PlaygroundSimulator.run()` already writes.

## Files outside role C

None yet.

## Fixes

- **Intermittent 500s from the API (phase 1).** FastAPI opened a request's SQLite connection in one worker thread and ran the endpoint in another, which SQLite rejects by default. About a third of board polls failed. `db.connect()` now passes `check_same_thread=False`; each connection still serves one request at a time. The worker is unaffected. Covered by `test_connection_works_across_threads`.

## Merge notes

- `web/src/App.tsx` now only holds state, polling and layout. The request list, request detail and candidate card live in `web/src/components/`. If you changed the old `Detail` or `CandidateView` functions in `App.tsx`, move that change into `RequestDetail.tsx` or `CandidateCard.tsx`.

## Needs from others

| What | Who | State |
|---|---|---|
| Keep Droid diffs in unified diff format (the board will color `+` and `-` lines) | B | Open |

## How to check

```sh
.venv/bin/python -m pytest -q
cd web && npm run build
```

Fake mode, with no GPU or API key: run `.venv/bin/rewardsmith-server` and `.venv/bin/rewardsmith-worker`, then `cd web && npm run dev` and open http://localhost:5173.
