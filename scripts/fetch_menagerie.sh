#!/usr/bin/env bash
# Playground's first env load git-clones all of mujoco_menagerie (slow, ~1 GB).
# We only need the bodies' robot models, so fetch just those folders at Playground's pinned commit:
#   unitree_go1  Go1JoystickFlatTerrain (locomotion/go1/base.py get_assets)
#   unitree_g1   G1JoystickFlatTerrain  (locomotion/g1/base.py get_assets: *.xml + assets/)
#   franka_emika_panda  PandaPickCube   (manipulation/franka_emika_panda/panda.py get_assets)
# MENAGERIE_MODELS="..." picks others. Safe to run again.
set -euo pipefail

PY=${PY:-.venv/bin/python}
MODELS=${MENAGERIE_MODELS:-unitree_go1 unitree_g1 franka_emika_panda}
DEST=$("$PY" -c "from mujoco_playground._src import mjx_env; print(mjx_env.MENAGERIE_PATH)")
SHA=$("$PY" -c "from mujoco_playground._src import mjx_env; print(mjx_env.MENAGERIE_COMMIT_SHA)")

missing=()
for m in $MODELS; do
  compgen -G "$DEST/$m/*.xml" >/dev/null && [ -d "$DEST/$m/assets" ] || missing+=("$m")
done
if [ ${#missing[@]} -eq 0 ]; then
  echo "menagerie already has $MODELS at $DEST"
  exit 0
fi

# Keep whatever an earlier sparse checkout already had (other sessions' bodies).
keep=""
if [ -d "$DEST/.git" ]; then
  keep=$(git -C "$DEST" sparse-checkout list 2>/dev/null || true)
fi
# shellcheck disable=SC2086  # word-split the model lists on purpose
paths=$(printf '%s\n' $MODELS $keep | sort -u)

TMP="$DEST.tmp"
rm -rf "$TMP" && git init -q "$TMP"
git -C "$TMP" remote add origin https://github.com/deepmind/mujoco_menagerie.git
# shellcheck disable=SC2086
git -C "$TMP" sparse-checkout set $paths
git -C "$TMP" fetch -q --depth 1 --filter=blob:none origin "$SHA"
git -C "$TMP" checkout -q FETCH_HEAD
rm -rf "$DEST" && mv "$TMP" "$DEST"
echo "fetched $(echo $paths) into $DEST"
