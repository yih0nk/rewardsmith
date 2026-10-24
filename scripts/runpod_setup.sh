#!/usr/bin/env bash
# One-time setup on a Runpod GPU pod (Ubuntu image with CUDA 12).
# Expose HTTP port 8000 on the pod; its proxy URL serves the board.
set -euo pipefail

apt-get update -y && apt-get install -y --no-install-recommends libegl1 libgl1 ffmpeg git
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -U "jax[cuda12]"
uv pip install --python .venv/bin/python -e ".[sim,dev]"

bash scripts/fetch_menagerie.sh

echo 'export MUJOCO_GL=egl' >> ~/.bashrc
export MUJOCO_GL=egl

.venv/bin/python -c "import jax; print('jax backend:', jax.default_backend())"
echo "Expect 'gpu' above. Next: build the web board (cd web && npm i && npm run build), then run server + worker."
