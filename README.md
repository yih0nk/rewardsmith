# Rewardsmith

Type a behavior ("make the robot dog walk backwards"). Rewardsmith writes the
reward function, trains a simulated quadruped on it, watches the rollout video,
and rewrites its own reward until the robot does what you asked.

Built from scratch at Cal Hacks 13.0 (Oct 23–25, 2026).

## How it works

```
request ──► checker (Claude)      pass/fail success check over rollout metrics, fixed before training
        ──► writer (Claude)       round 1: K candidate rewards
            editor (Droid/Claude)  round 2+: revise the best reward, with a diff
        ──► gate                   compile + short rollout, errors go back to the writer
        ──► train                  PPO in MuJoCo Playground on the GPU
        ──► render + measure       mp4 for the board, frames + metrics for the critic
        ──► check + critic (Claude) run the success check; critic watches frames, scores 0–10
                                   a request succeeds only if the check passes AND the score clears the bar
        ──► best candidate seeds the next round
```

Everything runs on one Runpod GPU pod: a FastAPI server (API + the board) and a
worker process share a SQLite file. The pod's proxy URL is the link you send judges.

## Layout

| Path | What | Owner |
|---|---|---|
| `rewardsmith/sim/playground.py` | Playground env, reward injection, training, render, metrics | A |
| `rewardsmith/agents/` | Claude writer and critic, Droid and Claude editors, fakes | B |
| `rewardsmith/orchestrator.py` | The worker loop | B |
| `rewardsmith/server.py`, `db.py` | API and storage | C |
| `web/` | The board (React + Vite) | C |
| `rewardsmith/bench/behaviors.py` | The 8 benchmark behaviors and their checks | D |

## Run it on a laptop (no GPU, no API key)

Fake mode returns canned rewards and fake metrics, so the loop and board work today.

```sh
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -e ".[dev]"
cp .env.example .env            # FAKE_SIM=1 and FAKE_LLM=1 by default

.venv/bin/rewardsmith-server    # API on :8000
.venv/bin/rewardsmith-worker    # second terminal
cd web && npm install && npm run dev   # board on :5173
```

Set `REWARDSMITH_FAKE_LLM=0` and `ANTHROPIC_API_KEY` to use real Claude calls
with the fake simulator.

## How Droid is used

From round 2 on, Factory Droid can revise the best rewards so far (`REWARDSMITH_EDITOR=droid`).

- **Trigger:** a finished round. For each reward being revised, the worker hands Droid that
  reward, the critic's notes, the rollout metrics and every earlier attempt.
- **Limits:** each revision runs in a fresh folder holding only those files, with the reward
  copied K times (`candidate_0.py` ...). Droid runs at low autonomy with shell commands
  disabled; it may edit files there, and anything beyond that is refused because no
  permission handler is attached.
- **Output:** we read back each candidate file and show its diff on the board, with Droid's
  one-line reason for the change. Untouched files are dropped.

Try it alone: `pip install -e ".[droid]"`, put `FACTORY_API_KEY` in `.env`, then
`.venv/bin/python scripts/try_droid.py`. The folders Droid saw stay under
`.rewardsmith/artifacts/droid/`. `REWARDSMITH_EDITOR=claude` (the default) uses Claude instead.

To try the reward writer alone with a real API key: `.venv/bin/python scripts/try_writer.py "walk backwards"`.

## Run it on Runpod

1. Start a GPU pod with a CUDA 12 image and expose HTTP port 8000.
2. Clone the repo, then `bash scripts/runpod_setup.sh`. It should print `jax backend: gpu`.
3. Build the board once: `cd web && npm install && npm run build`. The server serves `web/dist`.
4. `export REWARDSMITH_FAKE_SIM=0 REWARDSMITH_FAKE_LLM=0 ANTHROPIC_API_KEY=...` (or put them in `.env`).
   For demos, also set `REWARDSMITH_SUCCESS_SCORE=11` so every round runs instead of stopping at the first score of 8.
5. Run `rewardsmith-server` and `rewardsmith-worker` (tmux keeps them alive).
6. Open `https://<pod-id>-8000.proxy.runpod.net`.

## Tests

```sh
.venv/bin/python -m pytest -q
```

## Benchmark

`scripts/benchmark.py` runs the team plan's behaviors through the full loop (Claude and
training, like any request) and reports the four Devpost numbers: success rate after
round 1 vs. by the last round, median rounds to success, generated vs. hand-written
forward walking, and GPU minutes per learned behavior. It also reports how often the
critic's score agreed with the check.

```sh
python scripts/benchmark.py queue                    # on the pod, with the worker running
python scripts/benchmark.py report --wait --gpu-dollars-per-hour 0.69
```

Each behavior gets a fixed success check from `rewardsmith/bench/suite.py`, stored with
the request so the checker can't change it. Success means the check passed, not the
critic's score. `limp` and `pronk` need foot-contact metrics the rollout doesn't report
yet, so they are skipped and listed as unmeasured. Three other checks stand in for the
plan's (spin drift, run_fast fall rate, hind_stand pitch); the report says how.
Results go next to the manifest in `<REWARDSMITH_HOME>/benchmark/`.

## Quality regression suite

`scripts/regression.py` trains three fixed rewards (walk backwards, walk forward,
spin in place, all in `examples/rewards/`) with the `round` preset and seed 0,
through the same `PlaygroundSimulator.run_many` path the worker uses. For each it
records the final `episode_reward`, a behavior check from the rollout metrics
(e.g. `forward_velocity < -0.3` and `fell < 0.5`), `compile_s`, `train_s` and wall
time. A compare FAILs a case if its behavior check fails or its `episode_reward`
is more than 5% below `regression/baseline.json`. Timings are printed with their
change from the baseline but never fail the run.

Warp training isn't bit-deterministic: the same code and seed can end 10% lower, or
collapse outright, in a single run. So each case trains 3 times side by side
(`--repeats`), and it is judged on the median `episode_reward` and on the behavior
check passing in at least 2 of the 3 runs. The output lists every run's reward.
The walking cases repeat within 0.2 reward and gate the run. `spin_in_place` is
advisory: on unchanged code its training reward ranges from about 15 to 29, and a
whole set of 3 runs has fallen 2 out of 3 times. So a spin regression prints `WARN`
and the run still passes (`REGRESSION PASS (1 advisory warning)`). Look at it, but
don't block a PR on it alone.

To check that a branch (speed, robustness, ...) doesn't hurt quality, copy it to
your own folder on the GPU pod and run the compare under the shared GPU lock
(about 13 minutes). The copy has no `.git`, so pass your commit for the report:

```sh
rsync -a --exclude .venv --exclude .git --exclude node_modules --exclude .rewardsmith --exclude __pycache__ ./ pod:/workspace/<you>/
cd /workspace/<you>
flock /workspace/gpu.lock env MUJOCO_GL=egl REWARDSMITH_GIT_SHA=<your short sha> \
    /workspace/rewardsmith/.venv/bin/python scripts/regression.py --compare
```

The baseline (RTX 4090, warp) was recorded with the rollout sampling the policy's
actions, as brax's own eval does. A deterministic rollout makes the spin policy fall,
so `spin_in_place` fails its check on any branch without that render change.

It exits 1 and prints `REGRESSION FAIL` with the reasons if any case regressed.
`--only walk_forward` runs a subset. Videos, logs and `last_run.json` go to
`.rewardsmith/regression/`. The script puts its own checkout first on the import
path, for itself and the training subprocesses, so it tests your branch's code even
though the shared venv has `/workspace/rewardsmith` installed.

Re-record the baseline (`--record`, then commit `regression/baseline.json`) only
when a change is meant to move the numbers, such as a new preset or new
observations, and say so in the PR. `--record` refuses to write a baseline whose
behavior checks fail.

## Reward contract

Every generated reward is a module defining:

```python
def reward(state, action, prev_action) -> dict[str, jax.Array]:
    ...
```

It returns named terms. The trainer sums them, and the board shows them separately,
which makes reward hacking easy to spot.

## License

MIT
