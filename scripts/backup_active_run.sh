#!/usr/bin/env bash
# Mirror the ACTIVE training run to a directory outside the repo, every N seconds.
#
# Why this exists (2026-09-11): the run directory `2026-09-11_13-09-08_velocity`
# was deleted by mistake while training was still writing to it. Training died on
# `FileNotFoundError: .../events.out.tfevents...` (wandb's sync thread watches
# that file), and every checkpoint went with the directory — rsl_rl writes them
# into the run dir and wandb only stores SYMLINKS to them, so there was nothing
# to recover. 1h21m / 4273 iterations were one keystroke from being unrecoverable.
#
# The backup lives outside the repo on purpose: a repo cleanup must not be able
# to reach it.
#
# Usage:
#   scripts/backup_active_run.sh [interval_s] [dest_root]
#
# It follows the newest `logs/rsl_rl/*/*/` directory, so it keeps working when a
# run finishes and the next one starts.

set -u

INTERVAL="${1:-300}"
DEST_ROOT="${2:-/home/joyandai/microduck_rl_run_backup}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$DEST_ROOT"

echo "[backup] repo=$REPO dest=$DEST_ROOT interval=${INTERVAL}s"

while true; do
  ACTIVE="$(ls -dt "$REPO"/logs/rsl_rl/*/*/ 2>/dev/null | head -1)"
  if [ -n "${ACTIVE:-}" ] && [ -d "$ACTIVE" ]; then
    NAME="$(echo "${ACTIVE#$REPO/}" | tr '/' '_')"
    if rsync -a --delete \
        --exclude 'events.out.tfevents.*' \
        "$ACTIVE" "$DEST_ROOT/$NAME/" 2>/dev/null; then
      n=$(ls "$DEST_ROOT/$NAME"/model_*.pt 2>/dev/null | wc -l)
      echo "[backup] $(date +%H:%M:%S) $NAME -> $n checkpoints"
    else
      echo "[backup] $(date +%H:%M:%S) rsync failed for $NAME"
    fi
  else
    echo "[backup] $(date +%H:%M:%S) no active run dir"
  fi
  sleep "$INTERVAL"
done
