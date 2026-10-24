#!/usr/bin/env bash
# Playground's first env load git-clones all of mujoco_menagerie (slow, ~1 GB).
# We only need the Go1 model, so fetch just that folder at Playground's pinned commit.
set -euo pipefail

PY=${PY:-.venv/bin/python}
DEST=$("$PY" -c "from mujoco_playground._src import mjx_env; print(mjx_env.MENAGERIE_PATH)")
SHA=$("$PY" -c "from mujoco_playground._src import mjx_env; print(mjx_env.MENAGERIE_COMMIT_SHA)")

if [ -f "$DEST/unitree_go1/go1.xml" ]; then
  echo "menagerie already has unitree_go1 at $DEST"
  exit 0
fi

TMP="$DEST.tmp"
rm -rf "$TMP" && git init -q "$TMP"
git -C "$TMP" remote add origin https://github.com/deepmind/mujoco_menagerie.git
git -C "$TMP" sparse-checkout set unitree_go1
git -C "$TMP" fetch -q --depth 1 --filter=blob:none origin "$SHA"
git -C "$TMP" checkout -q FETCH_HEAD
rm -rf "$DEST" && mv "$TMP" "$DEST"
echo "fetched unitree_go1 into $DEST"
