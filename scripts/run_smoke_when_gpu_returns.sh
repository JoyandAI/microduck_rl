#!/usr/bin/env bash
# Wait for the GPU driver to come back, then run the mandated 64-env smoke test
# for the official-aligned velocity recipe + export its ONNX.
#
# Started 2026-09-17 ~16:35 after `nvidia-smi` began failing with
# "Failed to initialize NVML: Unknown Error" (the same transient crash recorded
# in docs/hd1910_defect_assessment_2026-09-17.md §5, which self-recovered in
# ~12 min). New CUDA processes cannot start while it is down; the already-running
# training holds its context and keeps iterating.
set -u
cd "$(dirname "$0")/.." || exit 1

LOG=logs/smoke_official_align.log
STATUS=logs/smoke_official_align.status
WAIT_MAX_S=${WAIT_MAX_S:-3600}
waited=0

: > "$LOG"
echo "waiting for GPU driver..." >> "$LOG"

while :; do
  if nvidia-smi --query-gpu=memory.free --format=csv,noheader >/dev/null 2>&1; then
    echo "[$(date +%H:%M:%S)] GPU back after ${waited}s" >> "$LOG"
    break
  fi
  if [ "$waited" -ge "$WAIT_MAX_S" ]; then
    echo "TIMEOUT after ${waited}s" > "$STATUS"
    exit 1
  fi
  sleep 20
  waited=$((waited + 20))
done

export UV_CACHE_DIR=/tmp/uv-cache WANDB_MODE=offline

echo "[$(date +%H:%M:%S)] SMOKE: 64 envs x 5 iters" >> "$LOG"
.venv/bin/python -m mjlab_microduck.train_cli \
  Mjlab-Velocity-Flat-MicroDuck \
  --env.scene.num-envs 64 --agent.max_iterations 5 \
  --agent.experiment_name smoke_official_align >> "$LOG" 2>&1
rc=$?
echo "[$(date +%H:%M:%S)] smoke rc=$rc" >> "$LOG"
[ $rc -ne 0 ] && { echo "SMOKE_FAIL rc=$rc" > "$STATUS"; exit $rc; }

RUN=$(ls -dt logs/rsl_rl/smoke_official_align/*/ 2>/dev/null | head -1)
CKPT=$(ls -t "$RUN"model_*.pt 2>/dev/null | head -1)
echo "[$(date +%H:%M:%S)] run=$RUN ckpt=$CKPT" >> "$LOG"
[ -z "$CKPT" ] && { echo "NO_CHECKPOINT" > "$STATUS"; exit 1; }

echo "[$(date +%H:%M:%S)] EXPORT" >> "$LOG"
.venv/bin/python scripts/export.py Mjlab-Velocity-Flat-MicroDuck \
  --checkpoint-file "$CKPT" --onnx-file "${RUN}smoke_official_align.onnx" >> "$LOG" 2>&1
rc=$?
echo "[$(date +%H:%M:%S)] export rc=$rc" >> "$LOG"
[ $rc -ne 0 ] && { echo "EXPORT_FAIL rc=$rc" > "$STATUS"; exit $rc; }

echo "OK run=$RUN onnx=${RUN}smoke_official_align.onnx" > "$STATUS"
echo "[$(date +%H:%M:%S)] DONE" >> "$LOG"
