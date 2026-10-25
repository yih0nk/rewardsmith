# Deploy on a fresh Runpod pod

A brand-new pod, any Runpod account, to a live board at `https://<pod-id>-8000.proxy.runpod.net`.
About 15 minutes of setup, then a 2-minute smoke test. Every command runs on the pod unless it says otherwise.

| Step | Time |
|---|---|
| 1. Create the pod | 2–4 min (until SSH answers) |
| 2. Clone | < 1 min |
| 3. `runpod_setup.sh` (apt, uv, jax CUDA, playground, menagerie) | 5–8 min |
| 4. Walkers | seconds |
| 5. Build the board (installs Node) | 1–2 min |
| 6. `.env` | < 1 min |
| 7. Start server + worker, GPU-priority daemon | < 1 min |
| 8. Smoke test: Instant "walk backwards" (Go1), then "walk forward on two legs" (G1) | ~2–4 min each (the first run compiles cold) |

## 1. Create the pod

Runpod console → **Pods → Deploy**:

- GPU: **RTX 4090** (24 GB), 1 GPU.
- Template: **Runpod PyTorch** (`runpod/pytorch:*-cu12*-ubuntu2404`; we ran `runpod/pytorch:1.4.1-cu1281-torch280-ubuntu2404`). Any CUDA 12+ image works.
- **Container disk 50 GB**, **volume disk ≥ 50 GB mounted at `/workspace`**. Only `/workspace` survives a restart.
- Edit the template's ports: **HTTP ports `8888,8000`**, **TCP port `22`**.
- Add your SSH public key under **Settings → SSH Public Keys** before deploying (or use the web terminal).

Or from a laptop with `runpodctl` signed in to that account:

```bash
runpodctl pod create --name rewardsmith --gpu-id "NVIDIA GeForce RTX 4090" --image runpod/pytorch:1.4.1-cu1281-torch280-ubuntu2404 --container-disk-in-gb 50 --volume-in-gb 50 --ports "8888/http,8000/http,22/tcp" --wait
```

Connect with the pod's **Connect → SSH over exposed TCP** command (`ssh root@<ip> -p <port> -i <your key>`).

## 2. Clone

`yih0nk/rewardsmith` is private: you need read access (ask Yihan), plus a GitHub token
(github.com → Settings → Developer settings → Personal access tokens → fine-grained, read-only "Contents" on this repo).

```bash
cd /workspace && git clone https://github.com/yih0nk/rewardsmith.git && cd rewardsmith
```

Git asks for a username (your GitHub username) and a password: paste the token. It isn't stored.

No GitHub access? From a laptop that has the repo: `rsync -a --exclude .venv --exclude node_modules --exclude web/dist --exclude .env -e "ssh -p <port> -i <key>" rewardsmith/ root@<ip>:/workspace/rewardsmith/`.

## 3. Python, JAX, Playground, robot models

```bash
cd /workspace/rewardsmith && bash scripts/runpod_setup.sh
```

It must end with `jax backend: gpu`. It also runs `scripts/fetch_menagerie.sh`, which fetches only the
`unitree_go1` and `unitree_g1` models from mujoco_menagerie (~15 s, instead of Playground cloning all ~1 GB on first use).

## 4. Pretrained walkers

```bash
bash scripts/fetch_walkers.sh
```

Copies `assets/walkers/{go1_walker,g1_walker}/params.pkl` (in git, checksummed) to
`/workspace/pretrained/<name>/params.pkl`, where the Go1 and G1 bodies read them. If it reports a checksum
failure, retrain instead (commands at the top of the script; Go1 ~5.5 min, G1 ~17 min on a 4090).

## 5. Build the board

```bash
bash scripts/pod/build_web.sh
```

The PyTorch image has no Node, so this installs Node 22 under `/workspace/.node`, then runs `npm ci && npm run build` in `web/`.

## 6. The API key

Type your own Anthropic key (it starts with `sk-ant-api`); this keeps it off the screen and out of shell history:

```bash
read -rsp "Anthropic API key: " K && printf 'ANTHROPIC_API_KEY=%s\n' "$K" > .env && unset K && echo
```

Never commit `.env` (it's in `.gitignore`). Optional extra line: `REWARDSMITH_SUCCESS_SCORE=11`
(every round runs instead of stopping at the first score of 8, for demos). The robot is picked per task in the board.

## 7. Start it

```bash
bash scripts/pod/start_live.sh
```

It kills any old server/worker, loads `.env` without printing it, and starts both with
`REWARDSMITH_HOME=/workspace/rs_live`, real sim, real Claude, `MUJOCO_GL=egl`,
`WARP_CACHE_PATH=/workspace/.cache/warp`. The last line should be
`{"ok":true,"fake_sim":false,"fake_llm":false}`. Logs: `/workspace/server.log`, `/workspace/worker.log`.
Run it again after any `git pull` or `.env` change.

Then the GPU-priority daemon (pauses other GPU jobs while the live board has a request running):

```bash
setsid nohup bash scripts/pod/gpu_priority.sh >/dev/null 2>&1 &
```

Other GPU work (benchmarks, walker training) goes through `bash scripts/pod/gpu_job.sh <command>` so it waits for the live board.

## 8. Smoke test

1. Open `https://<pod-id>-8000.proxy.runpod.net` (pod id from the Runpod console).
2. New task: type **walk backwards**. In the composer, turn the **Effort** switch **off** (Instant; new tasks
   default to Effort), set **Robot** to **Unitree Go1** (or leave Auto), send.
3. Within ~2–4 minutes the run should show a reward, training progress, a rollout video and a score.
   The first run on a new pod is the slowest (warp/JAX compile cold; later runs reuse `/workspace/.cache/warp`).
4. Second smoke test, the humanoid: new task **walk forward on two legs**, Effort off, **Robot: Unitree G1**.
   It should show the G1 walking in the video (G1 runs take longer than Go1's; it uses the G1 walker from step 4).
   **Auto** lets Claude pick the robot from the task text.

If it fails: `tail -50 /workspace/worker.log`. A `401 invalid x-api-key` means the `.env` key is wrong
(an OAuth token is not an API key): redo step 6 and step 7.

## After a pod restart

Container disk is wiped (apt packages, `~/.bashrc`); `/workspace` is kept. Run:

```bash
cd /workspace/rewardsmith && bash scripts/runpod_setup.sh && bash scripts/pod/start_live.sh
```

then start `gpu_priority.sh` again.
