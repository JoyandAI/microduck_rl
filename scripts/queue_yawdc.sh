#!/usr/bin/env bash
# Next arm, ONE variable away from the arm that is running now (agent A's prescription).
#
#   ours_head   = our recipe (12 deviations) + the head block          <- running
#   ours_yawdc  = ours_head + MICRODUCK_YAW_DC=1                        <- this script
#
# WHY YAW_DC (agent A, 2026-09-18, measured at the operating point):
#   The ONLY term pricing yaw in the default recipe is `track_angular_velocity`,
#   and against a persistent bias b it is quadratic: -0.543*b^2 per step. At the
#   measured per-env bias |b| = 0.1001 rad/s that is -0.0054/step = 0.080 % of the
#   total reward (6.749). Worse, |wz| (0.873) is 8.7x the bias, so the gradient
#   flips sign point-to-point and only 15.2 % of samples carry a usable signal.
#   `yaw_dc_rate_penalty` replaces -0.543*b^2 with -3.0*|EMA_0.8s(wz)| (L1 on a
#   filtered rate): at the same bias that is -0.300/step, 55.6x stronger, and the
#   marginal slope is 27.6x. Removing the bias pays back +0.290/step ~ 4.3 % of
#   total reward (it is worth 0.11 % today).
#
# HONEST ACCOUNTING (agent A): of the -0.455/step total charge, 0.156/step
# (2.3 %) is leakage tax on the gait's own yaw oscillation, and this term does
# NOT fix turning at all.
#
# PRE-REGISTERED FALSIFICATION (agent A): if |EMA| is still 0.10-0.15 rad/s about
# 1000 iterations after the 300-iteration ramp completes, then "the yaw channel is
# under-priced" is FALSIFIED and the remaining cause is reachability -- more weight
# will not help. Check it in the run's own log: the `Metrics/yaw_dc_abs` term
# (added 2026-09-18) reports exactly this quantity every iteration.
#
# Usage: nohup setsid bash scripts/queue_yawdc.sh > logs/queue_yawdc.out 2>&1 &
set -u

cd "$(dirname "$0")/.." || exit 1
OUT=logs/queue_yawdc
STATUS="$OUT/status.txt"
RESULTS="$OUT/results.tsv"
mkdir -p "$OUT"

ITERS=${ITERS:-9000}
ENVS=${ENVS:-2048}
SMOKE_ITERS=${SMOKE_ITERS:-5}
GPU_MIN_FREE=${GPU_MIN_FREE:-9000}
TASK="Mjlab-Velocity-Flat-MicroDuck"
LABEL=ours_yawdc
PRIOR_QUEUE_STATUS=logs/queue_ours/status.txt

export UV_CACHE_DIR=/tmp/uv-cache
export MPLCONFIGDIR=/tmp/matplotlib
export WANDB_MODE=offline
export MICRODUCK_TASK_NAMED_RUNS=1

OFF='MICRODUCK_STANCE_HIP_ROLL=0 MICRODUCK_ABDUCTION=0 MICRODUCK_YAW_SYMMETRY=0 MICRODUCK_SYMMETRY=0'
OURS='MICRODUCK_STANCE_HYGIENE=1 MICRODUCK_PER_JOINT_POSTURE=1 MICRODUCK_AIR_TIME_GATE=1'
OURS="$OURS MICRODUCK_HEADING_HOLD=1 MICRODUCK_HIP_ROLL_STD=0.12 MICRODUCK_POSE_ZERO_CMD_PROB=0.1"
OURS="$OURS MICRODUCK_HEAD_STABILITY=1 MICRODUCK_TURN_FIRST=0"
ARM_ENV="$OFF $OURS MICRODUCK_YAW_DC=1"
WANT="Episode_Reward/yaw_dc Episode_Reward/stance_hip_yaw Episode_Reward/head_motion Metrics/yaw_dc_abs"

log() { printf '[%(%F %T)T] %s\n' -1 "$*" | tee -a "$STATUS" >&2; }
fingerprint() {
  .venv/bin/python - <<'PY'
from mjlab_microduck.train_hook import actuator_params_fingerprint
print(actuator_params_fingerprint()[0])
PY
}
gpu_free_mib() {
  nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -dc '0-9'
}

log '========================================================================'
log "[$LABEL] queued behind the running queue (user: 不用全部同时开)"
log "[$LABEL] one variable from ours_head: + MICRODUCK_YAW_DC=1"
log '========================================================================'

# ── serial gate: prior queue finished AND no training anywhere ───────────────
waited=0
while :; do
  marked=0; busy=0
  grep -q "queue complete" "$PRIOR_QUEUE_STATUS" 2>/dev/null && marked=1
  pgrep -f "mjlab_microduck[.]train_cli" >/dev/null && busy=1
  if [ "$marked" -eq 1 ] && [ "$busy" -eq 0 ]; then
    log "[$LABEL] prior queue done and GPU idle after ${waited}s"
    break
  fi
  [ $((waited % 600)) -eq 0 ] && log "[$LABEL] waiting: prior_marker=${marked} training_busy=${busy} (${waited}s)"
  sleep 60
  waited=$((waited + 60))
done

free=$(gpu_free_mib || true)
while [ -z "$free" ] || [ "$free" -lt "$GPU_MIN_FREE" ]; do
  sleep 30; free=$(gpu_free_mib || true)
done
log "[$LABEL] GPU ready: ${free} MiB free"

# ── smoke + switch verification ─────────────────────────────────────────────
SLOG="$OUT/${LABEL}_smoke.log"
log "[$LABEL] smoke: 64 envs x ${SMOKE_ITERS}"
env $ARM_ENV MICRODUCK_RUN_DIR_NAME="${LABEL}_smoke" \
  .venv/bin/python -m mjlab_microduck.train_cli "$TASK" \
    --env.scene.num-envs 64 --agent.max_iterations "$SMOKE_ITERS" > "$SLOG" 2>&1
rc=$?
if [ "$rc" -ne 0 ] || grep -q 'Traceback (most recent call' "$SLOG"; then
  log "[$LABEL] SMOKE FAILED rc=${rc}"; exit 1
fi
missing=""
for t in $WANT; do grep -q "$t" "$SLOG" || missing="$missing $t"; done
[ -n "$missing" ] && { log "[$LABEL] SMOKE missing:$missing"; exit 1; }
grep -q "run directory named by task: ${LABEL}_smoke" "$SLOG" || {
  log "[$LABEL] FATAL: task-named run dirs did not engage"; exit 1; }
log "[$LABEL] smoke passed; verified:$WANT"

# ── train ───────────────────────────────────────────────────────────────────
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
{ [ "$rc" -ne 0 ] || [ ! -d "$run_dir" ]; } && { log "[$LABEL] TRAIN FAILED rc=${rc}"; exit 1; }
log "[$LABEL] training complete: ${run_dir}"

# ── the pre-registered falsification check, from the run's own log ───────────
log "[$LABEL] pre-registered check: Metrics/yaw_dc_abs after the ramp (target: < 0.10 rad/s)"
grep -o 'Metrics/yaw_dc_abs: [0-9.]*' "$TLOG" | tail -3 | sed 's/^/    /' | tee -a "$STATUS"

# ── export + P64 acceptance (same protocol as every other arm) ───────────────
ck=$(ls -t "$run_dir"/model_*.pt 2>/dev/null | head -1)
onnx="$OUT/${LABEL}.onnx"
.venv/bin/python scripts/export.py "$TASK" --checkpoint-file "$ck" \
  --onnx-file "$onnx" > "$OUT/${LABEL}_export.log" 2>&1 || { log "[$LABEL] export failed"; exit 1; }
log "[$LABEL] onnx=${onnx}"
.venv/bin/python scripts/eval_straightness.py --onnx "$onnx" --task "$TASK" \
  --lin-vel-x 0.2 --num-envs 64 --seed 0 --seconds 20 --freeze-pose-cmd \
  --json "$OUT/${LABEL}_walk.json" > "$OUT/${LABEL}_walk.log" 2>&1 || log "[$LABEL] walk eval failed"
.venv/bin/python scripts/check_posture_acceptance.py --onnx "$onnx" --keep-json \
  > "$OUT/${LABEL}_acceptance.log" 2>&1 || log "[$LABEL] acceptance has FAIL rows (expected; see log)"
log "[$LABEL] done"
