#!/usr/bin/env bash
# Third alignment arm — the CURRENT default recipe, i.e. official + all THREE
# head-stability terms (head_motion, head_droop, head_forward).
#
# Why it exists: the two arms launched at 16:49 predate `head_forward` (added
# 2026-09-17 ~17:00), so neither of them trains the recipe that is now the
# default. `head_forward` is the term the user asked for ("头要一直向前") and it
# is the only one of the three whose target behaviour is NOT already satisfied by
# the incumbent policies (measured: standing head_roll +0.119 rad = 6.8 deg vs
# the official's -0.035), so it is the one that needs an arm.
#
# VRAM: velocity @2048 = ~2.9 GiB (AGENTS.md, measured). The two running arms use
# ~6.0 GiB of 12.3, so this fits; it does slow all three (concurrency is not free).
#
# Usage: nohup setsid bash scripts/queue_align_arm3.sh > logs/queue_align_arm3.out 2>&1 &
set -u

cd "$(dirname "$0")/.." || exit 1
OUT=logs/queue_align
STATUS="$OUT/status.txt"
RESULTS="$OUT/results.tsv"

ITERS=${ITERS:-9000}
ENVS=${ENVS:-2048}
SMOKE_ITERS=${SMOKE_ITERS:-5}
GPU_MIN_FREE=${GPU_MIN_FREE:-3200}
GPU_WAIT_MAX_S=${GPU_WAIT_MAX_S:-86400}
LABEL=align_fwd
TASK="Mjlab-Velocity-Flat-MicroDuck"

export UV_CACHE_DIR=/tmp/uv-cache
export MPLCONFIGDIR=/tmp/matplotlib
export WANDB_MODE=offline
# Task-named run dirs are opt-in; without this the run lands in a timestamped
# `velocity` dir and two arms can collide (that killed the first launch).
export MICRODUCK_TASK_NAMED_RUNS=1

# Every legacy deviation pinned OFF; only the head block is on (it is the default,
# but pin it so the arm's identity does not depend on the ambient shell).
ARM_ENV='MICRODUCK_STANCE_HYGIENE=0 MICRODUCK_STANCE_HIP_ROLL=0 MICRODUCK_PER_JOINT_POSTURE=0'
ARM_ENV="$ARM_ENV MICRODUCK_AIR_TIME_GATE=0 MICRODUCK_HEADING_HOLD=0 MICRODUCK_YAW_HOLD_ZERO_CMD=0"
ARM_ENV="$ARM_ENV MICRODUCK_ABDUCTION=0 MICRODUCK_YAW_DC=0 MICRODUCK_YAW_SYMMETRY=0"
ARM_ENV="$ARM_ENV MICRODUCK_TURN_FIRST=0 MICRODUCK_SYMMETRY=0 MICRODUCK_HIP_ROLL_STD=0.05"
ARM_ENV="$ARM_ENV MICRODUCK_POSE_ZERO_CMD_PROB=0.0 MICRODUCK_HEAD_STABILITY=1"

log() { printf '[%(%F %T)T] %s\n' -1 "$*" | tee -a "$STATUS" >&2; }
fingerprint() {
  .venv/bin/python - <<'PY'
from mjlab_microduck.train_hook import actuator_params_fingerprint
print(actuator_params_fingerprint()[0])
PY
}

gpu_free_mib() {
  nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null |
    head -1 | tr -dc '0-9'
}

# ── SERIAL GATE ──────────────────────────────────────────────────────────────
# User instruction 2026-09-17: "排到队列里，不用全部同时开" — this arm goes into the
# queue BEHIND the two running arms instead of training alongside them. The repo's
# own recovery queue is deliberately serial for the same reason it states: running
# two velocity runs together roughly doubles the per-iteration time and makes an
# interrupted long run more likely. Wait for every other training to exit first.
gate=0
while pgrep -f "mjlab_microduck[.]train_cli" >/dev/null; do
  [ $((gate % 600)) -eq 0 ] && log "[$LABEL] queued: waiting for the running arms to finish (${gate}s)"
  sleep 60
  gate=$((gate + 60))
done
log "[$LABEL] GPU is free of other trainings after ${gate}s; starting serial run"

waited=0
while :; do
  free=$(gpu_free_mib || true)
  if [ -n "$free" ] && [ "$free" -ge "$GPU_MIN_FREE" ]; then
    log "[$LABEL] GPU ready: ${free} MiB free after ${waited}s"
    break
  fi
  [ $((waited % 300)) -eq 0 ] && log "[$LABEL] waiting for ${GPU_MIN_FREE} MiB free (${waited}s, free=${free:-?})"
  [ "$waited" -ge "$GPU_WAIT_MAX_S" ] && { log "[$LABEL] GPU wait exhausted"; exit 1; }
  sleep 30
  waited=$((waited + 30))
done

# ── smoke + switch verification ──────────────────────────────────────────────
SLOG="$OUT/${LABEL}_smoke.log"
log "[$LABEL] smoke: 64 envs x ${SMOKE_ITERS}"
env $ARM_ENV MICRODUCK_RUN_DIR_NAME="${LABEL}_smoke" \
  .venv/bin/python -m mjlab_microduck.train_cli "$TASK" \
    --env.scene.num-envs 64 --agent.max_iterations "$SMOKE_ITERS" > "$SLOG" 2>&1
rc=$?
if [ "$rc" -ne 0 ] || grep -q 'Traceback (most recent call' "$SLOG"; then
  log "[$LABEL] SMOKE FAILED rc=${rc}; see ${SLOG}"; exit 1
fi
for term in head_motion head_droop head_forward; do
  grep -q "Episode_Reward/${term}" "$SLOG" || {
    log "[$LABEL] FATAL: ${term} missing from the smoke reward log"; exit 1; }
done
grep -q "run directory named by task: ${LABEL}_smoke" "$SLOG" || {
  log "[$LABEL] FATAL: task-named run dirs did not engage"; exit 1; }
log "[$LABEL] smoke passed (all three head terms present, run-dir naming engaged)"

# ── train ────────────────────────────────────────────────────────────────────
FP_BEFORE=$(fingerprint)
TLOG="$OUT/${LABEL}_train.log"
log "[$LABEL] TRAIN start: ${ITERS} iters @ ${ENVS}; fingerprint=${FP_BEFORE}"
env $ARM_ENV MICRODUCK_RUN_DIR_NAME="$LABEL" \
  .venv/bin/python -m mjlab_microduck.train_cli "$TASK" \
    --env.scene.num-envs "$ENVS" --agent.max_iterations "$ITERS" \
    --agent.save_interval 500 > "$TLOG" 2>&1
rc=$?
FP_AFTER=$(fingerprint)
run_dir="logs/rsl_rl/velocity/$LABEL"
[ -d "$run_dir" ] || run_dir=$(grep -o 'Logging experiment in directory: .*' "$TLOG" | tail -1 | sed 's/.*directory: //')
printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
  "$LABEL" "$TASK" "$ITERS" "$ENVS" "$rc" "${run_dir:--}" "$FP_BEFORE" "$FP_AFTER" >> "$RESULTS"
[ "$FP_BEFORE" != "$FP_AFTER" ] && { log "[$LABEL] CONTAMINATED fingerprint"; exit 1; }
[ "$rc" -ne 0 ] && { log "[$LABEL] TRAIN FAILED rc=${rc}; see ${TLOG}"; exit 1; }
log "[$LABEL] training complete: ${run_dir}"

# ── export + the same P64 acceptance the other arms get ──────────────────────
ck=$(ls -t "$run_dir"/model_*.pt 2>/dev/null | head -1)
[ -n "$ck" ] || { log "[$LABEL] no checkpoint"; exit 1; }
onnx="$OUT/${LABEL}.onnx"
.venv/bin/python scripts/export.py "$TASK" --checkpoint-file "$ck" \
  --onnx-file "$onnx" > "$OUT/${LABEL}_export.log" 2>&1 || { log "[$LABEL] export failed"; exit 1; }
log "[$LABEL] onnx=${onnx}"

.venv/bin/python scripts/eval_straightness.py --onnx "$onnx" --task "$TASK" \
  --lin-vel-x 0.2 --num-envs 64 --seed 0 --seconds 20 --freeze-pose-cmd \
  --json "$OUT/${LABEL}_walk.json" > "$OUT/${LABEL}_walk.log" 2>&1 || log "[$LABEL] walk eval failed"
.venv/bin/python scripts/eval_straightness.py --onnx "$onnx" --task "$TASK" \
  --lin-vel-x 0.0 --num-envs 64 --seed 0 --seconds 20 --freeze-pose-cmd \
  --json "$OUT/${LABEL}_stand.json" > "$OUT/${LABEL}_stand.log" 2>&1 || log "[$LABEL] stand eval failed"
.venv/bin/python scripts/check_posture_acceptance.py --onnx "$onnx" --keep-json \
  > "$OUT/${LABEL}_acceptance.log" 2>&1 || log "[$LABEL] acceptance failed"
log "[$LABEL] done — evals in $OUT/${LABEL}_*.log"
