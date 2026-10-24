# Rewardsmith

Type a behavior ("make the robot dog walk backwards"). Rewardsmith writes the
reward function, trains a simulated quadruped on it, watches the rollout video,
and rewrites its own reward until the robot does what you asked.

Built from scratch at Cal Hacks 13.0 (Oct 23–25, 2026).

## How it works

```
request ──► writer (Claude)       round 1: K candidate rewards
            editor (Droid/Claude)  round 2+: revise the best reward, with a diff
        ──► gate                   compile + short rollout, errors go back to the writer
        ──► train                  PPO in MuJoCo Playground on the GPU
        ──► render + measure       mp4 for the board, frames + metrics for the critic
        ──► critic (Claude)        watches frames, scores 0–10, writes what to change
        ──► best candidate seeds the next round
```

Everything runs on one Runpod GPU pod: a FastAPI server (API + the board) and a
worker process share a SQLite file. The pod's proxy URL is the link you send judges.

## Layout

| Path | What | Owner |
|---|---|---|
| `rewardsmith/sim/playground.py` | Playground env, reward injection, training, render, metrics | A |
| `rewardsmith/agents/` | Claude writer and critic, editor (Droid TODO), fakes | B |
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

## Run it on Runpod

1. Start a GPU pod with a CUDA 12 image and expose HTTP port 8000.
2. Clone the repo, then `bash scripts/runpod_setup.sh`. It should print `jax backend: gpu`.
3. Build the board once: `cd web && npm install && npm run build`. The server serves `web/dist`.
4. `export REWARDSMITH_FAKE_SIM=0 REWARDSMITH_FAKE_LLM=0 ANTHROPIC_API_KEY=...`
5. Run `rewardsmith-server` and `rewardsmith-worker` (tmux keeps them alive).
6. Open `https://<pod-id>-8000.proxy.runpod.net`.

## Tests

```sh
.venv/bin/python -m pytest -q
```

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
