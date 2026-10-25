#!/usr/bin/env bash
# Put the pretrained walkers where the bodies look for them (steer/tune mode warm starts):
#   go1: /workspace/pretrained/go1_walker/params.pkl  (export.DEFAULT_WALKER, REWARDSMITH_WALKER)
#   g1:  /workspace/pretrained/g1_walker/params.pkl   (bodies/g1.py DEFAULT_WALKER, REWARDSMITH_G1_WALKER)
# Copies them from assets/walkers/ (in git, ~3.6 MB) after checking SHA256SUMS. Safe to run again.
# PRETRAINED=/some/dir puts them elsewhere (then point REWARDSMITH_WALKER / REWARDSMITH_G1_WALKER at it).
# If the copies are missing or corrupt, retrain instead (4090, warp: Go1 ~5.5 min, G1 ~17 min):
#   env MUJOCO_GL=egl .venv/bin/python scripts/train_walker.py --out /workspace/pretrained/go1_walker
#   env MUJOCO_GL=egl .venv/bin/python scripts/train_walker.py --env g1 --out /workspace/pretrained/g1_walker
set -euo pipefail
cd "$(dirname "$0")/.."

SRC=assets/walkers
DEST=${PRETRAINED:-/workspace/pretrained}

(cd "$SRC" && sha256sum -c --quiet SHA256SUMS 2>/dev/null || shasum -a 256 -c --quiet SHA256SUMS) || {
  echo "assets/walkers failed its checksum: retrain with scripts/train_walker.py (see the top of this script)" >&2
  exit 1
}

for name in go1_walker g1_walker; do
  mkdir -p "$DEST/$name"
  if cmp -s "$SRC/$name/params.pkl" "$DEST/$name/params.pkl"; then
    echo "$DEST/$name/params.pkl already up to date"
    continue
  fi
  if [ -e "$DEST/$name/params.pkl" ]; then
    mv "$DEST/$name/params.pkl" "$DEST/$name/params.pkl.bak"
    echo "kept the old $name as params.pkl.bak"
  fi
  cp "$SRC/$name/params.pkl" "$SRC/$name/walker.json" "$DEST/$name/"
  echo "installed $DEST/$name/params.pkl"
done
