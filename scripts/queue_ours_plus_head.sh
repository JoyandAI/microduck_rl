#!/usr/bin/env bash
# Back to OUR recipe (+ the head terms) — serial, two arms.
#
# User instruction 2026-09-18: "先不要按照官方的来了" (stop going by the official
# recipe for now) and "真机和仿真差不多，但是现在仿真问题比较多" (the real robot
# matches the sim; the problems are sim-side). Mass/CoM rework is therefore OFF
# the table and the remaining defects are the measured sim-side ones:
#
#   P1 steering is DEAD      track_angular_velocity 0.119/1.0; a ±34 deg/s yaw
#                            command moves the measured yaw rate <2%, wrong sign
#                            at a standstill. NOT under-priced — unreachable from
#                            the forward-shuffle basin. Never falsified, only
#                            aborted: turn-first curriculum (MICRODUCK_TURN_FIRST).
#   P2 hip_yaw parked/twist  hip_yaw at its limit 99.5% of the walk, dev L 25.8 /
#                            R 12.0 deg, abduction 18.7 deg, raw |a| 2.67 (max
#                            8.5 standing), crouched stance (trunk 86 mm).
#   P3 marches at standstill measured 6.7-20 mm of foot clearance (bar 8 mm).
#   P4 yaw unconstrained at zero command  same checkpoint drifts -17 to -158 deg
#                            across samples; heading_hold's speed gate zeroes the
#                            term when standing and nothing else prices yaw there.
#
# P2+P3+P4 are all inside ONE switch (MICRODUCK_STANCE_HYGIENE, which carries
# stance_hip_yaw + action_over_limit + standing_leg_vel + stillness_zero_cmd +
# no_stepping + yaw_hold_zero_cmd). P1 is a separate curriculum switch.
#
# Arm A: our old default recipe (all 12 deviations) + the head terms.
# Arm B: A + turn-first, so the steering question gets its own clean arm.
#
# Serial by user instruction ("不用全部同时开").
#
# Usage: nohup setsid bash scripts/queue_ours_plus_head.sh > logs/queue_ours.out 2>&1 &
set -u

cd "$(dirname "$0")/.." || exit 1
OUT=logs/queue_ours
STATUS="$OUT/status.txt"
RESULTS="$OUT/results.tsv"
mkdir -p "$OUT"

ITERS=${ITERS:-9000}
ENVS=${ENVS:-2048}
SMOKE_ITERS=${SMOKE_ITERS:-5}
GPU_MIN_FREE=${GPU_MIN_FREE:-9000}
GPU_WAIT_MAX_S=${GPU_WAIT_MAX_S:-86400}
TASK="Mjlab-Velocity-Flat-MicroDuck"

export UV_CACHE_DIR=/tmp/uv-cache
export MPLCONFIGDIR=/tmp/matplotlib
export WANDB_MODE=offline
export MICRODUCK_TASK_NAMED_RUNS=1   # opt-in; without it two arms share one dir

# Our measurements showed these make things worse or answer nothing, so they stay
# OFF even in "our recipe": single-joint hip_roll L1 (kills standing), abduction
# (same family), yaw-dc (never validated on a walking checkpoint), mirror loss.
OFF='MICRODUCK_STANCE_HIP_ROLL=0 MICRODUCK_ABDUCTION=0 MICRODUCK_YAW_DC=0'
OFF="$OFF MICRODUCK_YAW_SYMMETRY=0 MICRODUCK_SYMMETRY=0"
# Our recipe's default-ON deviations, restored.
OURS='MICRODUCK_STANCE_HYGIENE=1 MICRODUCK_PER_JOINT_POSTURE=1 MICRODUCK_AIR_TIME_GATE=1'
OURS="$OURS MICRODUCK_HEADING_HOLD=1 MICRODUCK_HIP_ROLL_STD=0.12 MICRODUCK_POSE_ZERO_CMD_PROB=0.1"
OURS="$OURS MICRODUCK_HEAD_STABILITY=1"
ARM_A="$OFF $OURS MICRODUCK_TURN_FIRST=0"
ARM_B="$OFF $OURS MICRODUCK_TURN_FIRST=1"

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
wait_for_gpu() {
  # serial: also require that no other training is running
  local waited=0 free
  while :; do
    if ! pgrep -f "mjlab_microduck[.]train_cli" >/dev/null; then
      free=$(gpu_free_mib || true)
      if [ -n "$free" ] && [ "$free" -ge "$GPU_MIN_FREE" ]; then
        log "GPU free of other trainings: ${free} MiB available after ${waited}s"
        return 0
      fi
    fi
    [ $((waited % 300)) -eq 0 ] && log "waiting for an idle GPU (${waited}s, free=${free:-?})"
    [ "$waited" -ge "$GPU_WAIT_MAX_S" ] && { log "GPU wait exhausted"; return 1; }
    sleep 30
    waited=$((waited + 30))
  done
}

# Smoke + SWITCH VERIFICATION: the arm's identity is only real if the terms it
# claims to enable actually appear in the reward/curriculum tables.
smoke_arm() {
  local label="$1" arm_env="$2" want="$3" log_file="$OUT/${label}_smoke.log"
  log "[$label] smoke: 64 envs x ${SMOKE_ITERS}"
  env $arm_env MICRODUCK_RUN_DIR_NAME="${label}_smoke" \
    .venv/bin/python -m mjlab_microduck.train_cli "$TASK" \
      --env.scene.num-envs 64 --agent.max_iterations "$SMOKE_ITERS" > "$log_file" 2>&1
  local rc=$?
  { [ "$rc" -ne 0 ] || grep -q 'Traceback (most recent call' "$log_file"; } && {
    log "[$label] SMOKE FAILED rc=${rc}"; return 1; }
  local missing=""
  for term in $want; do
    grep -q "$term" "$log_file" || missing="$missing $term"
  done
  [ -n "$missing" ] && { log "[$label] SMOKE: missing from the log:$missing"; return 1; }
  grep -q "run directory named by task: ${label}_smoke" "$log_file" || {
    log "[$label] FATAL: task-named run dirs did not engage"; return 1; }
  log "[$label] smoke passed; verified present:$want"
}

run_arm() {
  local label="$1" arm_env="$2" want="$3"
  wait_for_gpu || return 1
  smoke_arm "$label" "$arm_env" "$want" || return 1

  local fp_before tlog="$OUT/${label}_train.log"
  fp_before=$(fingerprint)
  log "[$label] TRAIN start: ${ITERS} iters @ ${ENVS}; fingerprint=${fp_before}"
  env $arm_env MICRODUCK_RUN_DIR_NAME="$label" \
    .venv/bin/python -m mjlab_microduck.train_cli "$TASK" \
      --env.scene.num-envs "$ENVS" --agent.max_iterations "$ITERS" \
      --agent.save_interval 500 > "$tlog" 2>&1
  local rc=$? fp_after run_dir="logs/rsl_rl/velocity/$label"
  fp_after=$(fingerprint)
  [ -d "$run_dir" ] || run_dir=$(grep -o 'Logging experiment in directory: .*' "$tlog" | tail -1 | sed 's/.*directory: //')
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$label" "$TASK" "$ITERS" "$ENVS" "$rc" "${run_dir:--}" "$fp_before" "$fp_after" >> "$RESULTS"
  [ "$fp_before" != "$fp_after" ] && { log "[$label] CONTAMINATED fingerprint"; return 1; }
  { [ "$rc" -ne 0 ] || [ ! -d "$run_dir" ]; } && { log "[$label] TRAIN FAILED rc=${rc}"; return 1; }
  log "[$label] training complete: ${run_dir}"

  local ck onnx="$OUT/${label}.onnx"
  ck=$(ls -t "$run_dir"/model_*.pt 2>/dev/null | head -1)
  .venv/bin/python scripts/export.py "$TASK" --checkpoint-file "$ck" \
    --onnx-file "$onnx" > "$OUT/${label}_export.log" 2>&1 || { log "[$label] export failed"; return 1; }
  log "[$label] onnx=${onnx}"

  .venv/bin/python scripts/eval_straightness.py --onnx "$onnx" --task "$TASK" \
    --lin-vel-x 0.2 --num-envs 64 --seed 0 --seconds 20 --freeze-pose-cmd \
    --json "$OUT/${label}_walk.json" > "$OUT/${label}_walk.log" 2>&1 || log "[$label] walk eval failed"
  .venv/bin/python scripts/eval_straightness.py --onnx "$onnx" --task "$TASK" \
    --lin-vel-x 0.0 --num-envs 64 --seed 0 --seconds 20 --freeze-pose-cmd \
    --json "$OUT/${label}_stand.json" > "$OUT/${label}_stand.log" 2>&1 || log "[$label] stand eval failed"
  .venv/bin/python scripts/check_posture_acceptance.py --onnx "$onnx" --keep-json \
    > "$OUT/${label}_acceptance.log" 2>&1 || log "[$label] acceptance table has FAIL rows (expected; see log)"
  log "[$label] done"
}

log '========================================================================'
log "OUR recipe + head terms; serial 2 arms; fingerprint=$(fingerprint)"
log "A=ours_head (all 12 deviations + head block)  B=ours_turn (A + turn-first)"
log '========================================================================'
printf 'label\ttask\titers\tenvs\trc\trun_dir\tfp_before\tfp_after\n' > "$RESULTS"

# Switch-verification targets. Only POSITIVE checks (grep for a thing being
# present) — "turn_first" must appear in arm B's curriculum table and must NOT
# appear in arm A's, but arm A's absence is not asserted by grep because the
# string can show up in a dumped config; the run's git diff records the env.
WANT_A="Episode_Reward/stance_hip_yaw Episode_Reward/stillness_zero_cmd Episode_Reward/yaw_hold_zero_cmd Episode_Reward/standing_leg_vel Episode_Reward/heading_hold Episode_Reward/head_motion Episode_Reward/head_droop Episode_Reward/head_forward"
WANT_B="Episode_Reward/stance_hip_yaw Episode_Reward/stillness_zero_cmd Episode_Reward/head_forward turn_first"

run_arm ours_head "$ARM_A" "$WANT_A" || log "[ours_head] FAILED"
run_arm ours_turn "$ARM_B" "$WANT_B" || log "[ours_turn] FAILED"

log "queue complete; results: $RESULTS"
