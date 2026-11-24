#!/usr/bin/env bash
# Adopt a training that lost its driver: wait for it to finish, then do the
# export + P64 evals + acceptance that its queue would have done.
#
# Why this exists: the driver script for `ours_head` was stopped on purpose (it
# would have auto-started a turn arm whose base agent F's evidence says is wrong),
# and SIGTERM to the parent bash does NOT kill the running python child. So the run
# keeps producing checkpoints with nobody to evaluate them.
#
# Usage: bash scripts/adopt_orphan_run.sh <label> <run_dir>
set -u
cd "$(dirname "$0")/.." || exit 1
LABEL="$1"; RUN_DIR="$2"
OUT=${OUT:-logs/queue_overnight}
TASK="Mjlab-Velocity-Flat-MicroDuck"
mkdir -p "$OUT"
export UV_CACHE_DIR=/tmp/uv-cache MPLCONFIGDIR=/tmp/matplotlib

log() { printf '[%(%F %T)T] %s\n' -1 "[adopt:$LABEL] $*" | tee -a "$OUT/status.txt" >&2; }

[ -d "$RUN_DIR" ] || { log "no run dir $RUN_DIR"; exit 1; }

# Wait for the orphaned training to exit (it is the only train_cli left by now).
waited=0
while pgrep -f "mjlab_microduck[.]train_cli" >/dev/null; do
  [ $((waited % 300)) -eq 0 ] && log "waiting for the orphaned training to finish (${waited}s)"
  sleep 30; waited=$((waited + 30))
done
log "training finished after ${waited}s; evaluating"

ck=$(ls -t "$RUN_DIR"/model_*.pt 2>/dev/null | head -1)
[ -n "$ck" ] || { log "no checkpoint in $RUN_DIR"; exit 1; }
onnx="$OUT/${LABEL}.onnx"
.venv/bin/python scripts/export.py "$TASK" --checkpoint-file "$ck" \
  --onnx-file "$onnx" > "$OUT/${LABEL}_export.log" 2>&1 || { log "export failed"; exit 1; }
log "exported $onnx from $(basename "$ck")"

for vx in 0.2 0.0; do
  .venv/bin/python scripts/eval_straightness.py --onnx "$onnx" --task "$TASK" \
    --lin-vel-x "$vx" --num-envs 64 --seed 0 --seconds 20 --freeze-pose-cmd \
    --json "$OUT/${LABEL}_vx${vx}.json" > "$OUT/${LABEL}_vx${vx}.log" 2>&1 ||
    log "eval vx=$vx failed"
done
.venv/bin/python scripts/gait_forensics.py --onnx "$onnx" --task "$TASK" \
  --lin-vel-x 0.2 --warmup 2 --seconds 12 --head-cmd zero \
  --json-out "$OUT/${LABEL}_gait.json" > "$OUT/${LABEL}_gait.log" 2>&1 ||
  log "gait forensics failed"
.venv/bin/python scripts/check_posture_acceptance.py --onnx "$onnx" --keep-json \
  > "$OUT/${LABEL}_acceptance.log" 2>&1 || log "acceptance has FAIL rows (expected)"
log "done"
