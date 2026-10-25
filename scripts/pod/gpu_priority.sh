#!/bin/bash
# Live-first GPU priority for the Rewardsmith pod.
#
# While the live board (REWARDSMITH_HOME=/workspace/rs_live, port 8000) has a request
# queued or running, every other Rewardsmith python process (benchmarks, other
# sessions' workers and training children) is paused with SIGSTOP, and resumed with
# SIGCONT when the live board goes idle. /workspace/live.busy exists while live is busy,
# so gpu_job.sh can hold new jobs back. Pause windows are logged to
# /workspace/gpu_priority.log so benchmark timings that overlap one can be discarded.
#
# Run: setsid nohup bash scripts/pod/gpu_priority.sh >/dev/null 2>&1 &
set -u
LIVE_HOME=/workspace/rs_live
API=http://localhost:8000/api/requests
MARK=/workspace/live.busy
LOG=/workspace/gpu_priority.log
STATE=/workspace/gpu_priority.paused
PIDFILE=/workspace/gpu_priority.pid

echo $$ > "$PIDFILE"
log() { echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }

resume_all() {
  if [ -s "$STATE" ]; then
    while read -r pid; do kill -CONT "$pid" 2>/dev/null; done < "$STATE"
    log "resumed $(wc -l < "$STATE") process(es)"
  fi
  : > "$STATE"
  rm -f "$MARK"
}
trap 'resume_all; log "daemon stopped"; rm -f "$PIDFILE"; exit 0' INT TERM EXIT

live_busy() {
  # Busy = any live request that is queued or training. Anything unreadable counts as idle.
  # `touch /workspace/live.hold` holds the GPU for the live board by hand (e.g. during a demo).
  [ -e /workspace/live.hold ] && return 0
  curl -s -m 5 "$API" | python3 -c '
import json, sys
try:
    rows = json.load(sys.stdin)
except Exception:
    sys.exit(1)
busy = {"queued", "running", "pausing", "cancelling"}
sys.exit(0 if any(r.get("status") in busy for r in rows) else 1)'
}

is_live() {  # the live server, worker and their children carry REWARDSMITH_HOME=rs_live
  tr '\0' '\n' < "/proc/$1/environ" 2>/dev/null | grep -qx "REWARDSMITH_HOME=$LIVE_HOME"
}

targets() {
  # Rewardsmith python processes that aren't the live board's, aren't web servers and aren't Jupyter.
  # Python interpreters only: shells, flock and timeout wrappers just wait on them.
  for pid in $(pgrep -f python); do
    [ "$pid" = "$$" ] && continue
    case "$(readlink "/proc/$pid/exe" 2>/dev/null)" in *python*) ;; *) continue ;; esac
    cmd=$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null) || continue
    case "$cmd" in
      *jupyter*|*uvicorn*|*rewardsmith-server*) continue ;;
    esac
    case "$cmd" in
      *rewardsmith*|*/workspace/wt-*|*scripts/*|*/workspace/rs_*) ;;
      *) continue ;;
    esac
    is_live "$pid" && continue
    echo "$pid"
  done
}

log "daemon started (pid $$)"
: > "$STATE"
paused=0
while true; do
  if live_busy; then
    if [ "$paused" = 0 ]; then
      touch "$MARK"
      log "live busy: pausing other GPU work"
      paused=1
    fi
    # Re-scan every tick so jobs started during a live run get paused too.
    for pid in $(targets); do
      if ! grep -qx "$pid" "$STATE"; then
        kill -STOP "$pid" 2>/dev/null && echo "$pid" >> "$STATE" && \
          log "paused $pid $(tr '\0' ' ' < /proc/$pid/cmdline 2>/dev/null | cut -c1-120)"
      fi
    done
  elif [ "$paused" = 1 ]; then
    log "live idle"
    resume_all
    paused=0
  fi
  sleep 2
done
