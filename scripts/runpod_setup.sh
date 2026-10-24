#!/usr/bin/env bash
# Set up (or, after a pod restart, repair) the Runpod GPU pod. Safe to run again.
# Clone the repo under /workspace so it survives restarts, and expose HTTP port 8000.
set -euo pipefail

if [ "$(uname -s)" = "Darwin" ]; then
  echo "This sets up the Runpod GPU pod. Run it on the pod, not your Mac: it would replace .venv." >&2
  exit 1
fi
cd "$(dirname "$0")/.."

# A pod restart wipes the container disk; only /workspace survives. Keep uv's Python
# and warp's compiled kernels there. The uv cache reached 9 GB and filled the volume,
# so don't keep one.
export UV_NO_CACHE=1
if [ -d /workspace ]; then
  export UV_PYTHON_INSTALL_DIR=/workspace/.uv/python
  export WARP_CACHE_PATH=/workspace/.cache/warp
fi
export MUJOCO_GL=egl

# System packages live on the container disk: reinstall after every restart.
apt-get update -y && apt-get install -y --no-install-recommends libegl1 libgl1 ffmpeg git
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# Reuse the venv only if its key packages import. A restart or pod migration can leave
# a half-copied venv whose Python still starts (the migration lost ml_dtypes metadata).
if ! .venv/bin/python -c "import jax, brax, mujoco, mujoco_playground, warp" 2>/dev/null; then
  uv venv --clear --python 3.11 .venv
fi
uv pip install --python .venv/bin/python -U "jax[cuda12]"
uv pip install --python .venv/bin/python -e ".[sim,dev]"

bash scripts/fetch_menagerie.sh

# Same settings for new shells (~/.bashrc is wiped on restart too, so this re-adds them).
for line in "export MUJOCO_GL=egl" "export UV_NO_CACHE=1" \
            ${UV_PYTHON_INSTALL_DIR:+"export UV_PYTHON_INSTALL_DIR=$UV_PYTHON_INSTALL_DIR"} \
            ${WARP_CACHE_PATH:+"export WARP_CACHE_PATH=$WARP_CACHE_PATH"}; do
  grep -qxF "$line" ~/.bashrc 2>/dev/null || echo "$line" >> ~/.bashrc
done

.venv/bin/python -c "import jax; print('jax backend:', jax.default_backend())"
echo "Expect 'gpu' above. Next: build the web board (cd web && npm i && npm run build), then run server + worker."
