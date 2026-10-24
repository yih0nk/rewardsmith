# Status: `role-c/3-hall-of-fame`

As of Sat Oct 24, 2026. Read this before working on this branch. For the short version a reviewer or merger needs (contracts, files touched outside role C, conflict spots), see [role-c-catchup.md](role-c-catchup.md).

## Where things stand

Role C (API, SQLite store, board) has built everything the team plan asks of it except what depends on other roles. The hall of fame works end to end, the board covers the plan's diff, round-comparison and QR items, and a review pass fixed every bug found in role C's files. The branch is tested in fake mode (no GPU, no API key). The real GPU and Claude paths have not been run from this branch.

## How this branch is built

```
main ── (merged in at 0098477)
  └─ codex/frontend-dashboard-draft   Corey Dai's dashboard, itself built on role-c/1-board-structure
       └─ role-c/3-hall-of-fame       this branch
```

- **Merge order:** Corey's branch goes into `main` first. Until then, a PR from this branch also shows Corey's 4 commits.
- **Role C commits on this branch**, oldest first:
  1. `92977a7` fix: API requests use their SQLite connection across threads
  2. `a5368e8` refactor: split the board into components, reserve `#hall`
  3. `745bec7` feat: hall-of-fame API, restart recovery, fake reward terms
  4. `d3dc244` feat: hall-of-fame page, flagging, colored diffs, round gallery, QR
  5. `8983adc` fix: reject blank behaviors, load `.env` only in entry points, favicon
- **Conventions:**
  - Never commit to `main`. Each phase gets a branch off the latest `origin/main`.
  - Commits and PRs carry no Co-Authored-By or other AI attribution.
  - Update `role-c-catchup.md` with every role C change.

## Decisions, and why

| Decision | Why |
|---|---|
| Reward terms are metrics named `term_<name>`, not a new field or column | A's `PlaygroundSimulator.run()` already writes them that way, and the orchestrator already saves `metrics`. So `orchestrator.py`, `RunResult` and the schema stay untouched. |
| The hall of fame uses manual flags with a note | The team plan names a hall of fame but not how runs get in. Manual flags fit entirely in role C's files. The critic already judges reward hacking, but the orchestrator discards that verdict, and storing it would mean changing B's code (see Open issues). |
| `flagged_at` is a timestamp, not a 0/1 flag | One column records whether a candidate is flagged and also orders the hall newest-first. |
| Interrupted runs are marked `failed` at worker startup, not re-queued | A re-queued request restarts at round 1 and would duplicate round numbers next to its partial rounds. Failing is honest, and the event tells the user to resubmit. |
| `.env` is loaded by the server module and the worker's `main()`, not by `config.load()` | Loading it in `config.load()` let a developer's `.env` leak into tests. Loading it in `server.main()` was too late, because the installed `rewardsmith-server` script imports the module first. |
| `check_same_thread=False` on SQLite connections | FastAPI opens a request's connection in one worker thread and may use it in another. Each connection still serves one request at a time. |
| Charts use the dashboard's CSS bars, not Recharts | The plan picked Recharts before Corey built the dashboard. Corey's round tiles and bars already show the score per round and the reward terms, so a chart library would duplicate them. |
| The round gallery appears only when 2 or more rounds have video | It exists for the plan's "round 1 vs. round 5" comparison. One video has nothing to compare against. |
| The QR code is on the home view and hidden at 650 px and below | Judges scan it from a laptop or projector screen. On a phone it's pointless. |
| The `package-lock.json` diff is limited to the new package | Newer npm versions drop `"peer": true` lines from the whole lockfile, which creates noise and merge conflicts. The lockfile was rebuilt from the committed one with only `qrcode.react` added. |

## Discoveries and gotchas

- **Runs can stop after round 1.** The worker stops once a candidate scores `REWARDSMITH_SUCCESS_SCORE` (default 8). For demos, set it to 11 so every round runs.
- **Cards show `written` for the whole GPU training run.** Candidates are now trained in parallel, after all of them pass the compile check, and the status only changes when training ends. The board's status line shows the latest event, "Training N candidate(s)", so users still see progress.
- **`fail_interrupted()` assumes one worker.** Starting a second worker would fail the first worker's running request.
- **The fake simulator makes no video.** The round gallery and hall-of-fame videos are covered by frontend tests but have not been seen in a browser with real footage.
- **Safari video already works.** `/artifacts` answers range requests (`206 Partial Content`), which Safari needs. No custom route is required.
- **`/artifacts` is public and serves more than videos:** also `reward.py`, `train.log` and `result.json` (which can hold tracebacks). Fine for a hackathon; there are no secrets in them.
- **Every API request opens a new SQLite connection** and re-runs the schema and migration check. Each open board polls every 1.5–2 s, so that's several connections a second per viewer. It's cheap at demo scale, but it's the first place to look if the API ever feels slow.

## Open issues

| Issue | Owner | Detail |
|---|---|---|
| A critic or Claude API error kills the whole run | B | In `run_request`, an exception from `critic.review(...)` fails the request. The candidate being scored stays `trained`, and the rest of the round and all later rounds are lost. Fix: wrap the call in try/except, mark that candidate `failed` with the error, and continue, the way crashed training runs are already handled. |
| The critic's reward-hacking verdict is discarded | B | `ClaudeCritic` gets a `reward_hacking` yes/no and only adds it to the critique text. Storing it would let the hall of fame show "critic suspects hacking". |
| Droid editor isn't wired in | B | `DroidEditor.revise` raises `NotImplementedError`. Revisions use `ClaudeEditor`. |
| 7 of the 8 benchmark checks can never pass | A | `bench/behaviors.py` reads metrics the rollout doesn't produce: `abs_forward_velocity`, `abs_yaw_rate`, `base_drift_m`, `fall_rate`, `height_ratio`, `contact_fr`, `airborne_all`, `pitch_deg`, `hold_s`. |
| No benchmark runner, and nothing computes the four Devpost numbers | Role D (unowned) | Queue the 8 behaviors, run the checks, compute the success rates, median rounds to success, comparison against the hand-written reward, and GPU cost. |
| This branch isn't merged | Role C | Merge Corey's branch first, then open the PR from this branch. |

## Not verified

- **GPU training and rendering** (A's code): its tests skip without a GPU, so nothing on this branch has trained a real policy.
- **Real Claude calls:** the request shape matches the current API reference, and the installed SDK (2.3.0) accepts every parameter, but nothing has been sent to the API.
- **New components at phone width:** the flag control, hall-of-fame cards and gallery were checked only at 1280 px wide.

## How to run and check

```sh
uv venv --python 3.11 .venv && uv pip install --python .venv/bin/python -e ".[dev]"
cp .env.example .env                      # fake simulator and fake LLM by default
.venv/bin/python -m pytest -q             # 17 passed, 4 skipped (GPU)
cd web && npm ci && npm run build && npm test   # 11 passed
```

Fake mode: `.venv/bin/rewardsmith-server` and `.venv/bin/rewardsmith-worker` from the repo root, then `cd web && npm run dev` and open http://localhost:5173.
