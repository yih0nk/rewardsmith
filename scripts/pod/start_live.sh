#!/usr/bin/env bash
# (Re)start the live board on the pod: server (API + board, port 8000) and worker, real sim and real Claude.
# Kills only the old rewardsmith-server/worker, loads .env without printing it, data in REWARDSMITH_HOME.
#   bash scripts/pod/start_live.sh
# Overrides: REWARDSMITH_HOME (default /workspace/rs_live), REWARDSMITH_ROUNDS (default 2), LOG_DIR (default /workspace).
set -euo pipefail
cd "$(dirname "$0")/../.."

if [ ! -s .env ] || ! grep -q '^ANTHROPIC_API_KEY=.\+' .env; then
  echo "$(pwd)/.env needs a line ANTHROPIC_API_KEY=sk-ant-... (see docs/deploy.md)" >&2
  exit 1
fi
[ -d web/dist ] || { echo "web/dist is missing: run bash scripts/pod/build_web.sh first" >&2; exit 1; }

for p in $(pgrep -f "bin/rewardsmith-(server|worker)$" || true); do kill "$p"; done
sleep 3

set -a; . ./.env; set +a
LOG_DIR=${LOG_DIR:-/workspace}
export REWARDSMITH_HOME=${REWARDSMITH_HOME:-/workspace/rs_live} \
  REWARDSMITH_ROUNDS=${REWARDSMITH_ROUNDS:-2} \
  REWARDSMITH_FAKE_SIM=0 REWARDSMITH_FAKE_LLM=0 \
  MUJOCO_GL=egl WARP_CACHE_PATH=/workspace/.cache/warp
mkdir -p "$REWARDSMITH_HOME" "$WARP_CACHE_PATH" "$LOG_DIR"

nohup setsid .venv/bin/rewardsmith-server >> "$LOG_DIR/server.log" 2>&1 < /dev/null &
nohup setsid .venv/bin/rewardsmith-worker >> "$LOG_DIR/worker.log" 2>&1 < /dev/null &
sleep 6
pgrep -af "bin/rewardsmith-(server|worker)$" || { echo "server/worker did not start: see $LOG_DIR/server.log, worker.log" >&2; exit 1; }
curl -s localhost:8000/api/health; echo
