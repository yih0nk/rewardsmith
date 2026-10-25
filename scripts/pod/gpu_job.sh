#!/bin/bash
# Run a GPU job on the Rewardsmith pod without slowing the live board.
# Usage: bash scripts/pod/gpu_job.sh <command> [args...]
# Waits while the live board is busy (/workspace/live.busy), then takes /workspace/gpu.lock.
# gpu_priority.sh may still pause the job (SIGSTOP) if a live request starts mid-run:
# check /workspace/gpu_priority.log before trusting wall-clock timings.
while [ -e /workspace/live.busy ]; do sleep 5; done
exec flock /workspace/gpu.lock "$@"
