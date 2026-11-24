#!/usr/bin/env bash
# Official-alignment queue — 2026-09-17.
#
# Two arms, both on the CURRENT 7.5 V HD-1910 physics, both with every legacy
# recipe deviation explicitly pinned OFF (= the official upstream recipe):
#
#   align_full      official recipe + the head-stability block (the user's ask)
#   align_official  official recipe verbatim (MICRODUCK_HEAD_STABILITY=0)
#
# WHY THESE TWO: the one experiment that separates "our recipe edits" from "our
# hardware" was run on 2026-09-15 (night_D_s42/s43/s44) but on the PRE-2026-09-16
# servo parameters (friction_base 0.01355 vs 0.00675 today, load_friction_motor
# 0.0164 vs 0.2986, kd 0.346 vs 0.5, max_velocity 8.03 vs 12.04) — so it cannot
# answer today's question. See docs/official_alignment_2026-09-17.md §3.
#
# The second arm exists so the head terms can be attributed at all: if the two
# arms differ on the head metrics, it is the head block; if they differ on
# nothing, the terms are free.
#
# Usage:
#   nohup setsid bash scripts/queue_official_align.sh > logs/queue_align.out 2>&1 &
set -u

cd "$(dirname "$0")/.." || exit 1
ROOT="$PWD"
OUT="$ROOT/logs/queue_align"
STATUS="$OUT/status.txt"
RESULTS="$OUT/results.tsv"
mkdir -p "$OUT"

ITERS=${ITERS:-9000}
ENVS=${ENVS:-2048}
SMOKE_ITERS=${SMOKE_ITERS:-5}
GPU_MIN_FREE=${GPU_MIN_FREE:-8000}
GPU_WAIT_MAX_S=${GPU_WAIT_MAX_S:-86400}

export UV_CACHE_DIR=/tmp/uv-cache
export MPLCONFIGDIR=/tmp/matplotlib
export WANDB_MODE=offline
# Task-named run directories are OPT-IN (train_cli._install_task_named_run_dirs
# returns immediately unless this is 1) and MICRODUCK_RUN_DIR_NAME only supplies
# the slug — without the toggle it is a silent no-op. That cost this queue one
# launch: both arms fell back to the timestamped `velocity` directory, landed on
# the same second, and died on a missing tfevents file 44 s in. Two arms must
# never share a log dir.
export MICRODUCK_TASK_NAMED_RUNS=1

TASK="Mjlab-Velocity-Flat-MicroDuck"
RUN_ROOT="logs/rsl_rl/velocity"

# Every deviation at its OFFICIAL value, pinned so a stray shell export cannot
# contaminate an arm (and so the arm's identity is self-documenting).
ALIGN_OFF='MICRODUCK_STANCE_HYGIENE=0 MICRODUCK_STANCE_HIP_ROLL=0 MICRODUCK_PER_JOINT_POSTURE=0'
ALIGN_OFF="$ALIGN_OFF MICRODUCK_AIR_TIME_GATE=0 MICRODUCK_HEADING_HOLD=0 MICRODUCK_YAW_HOLD_ZERO_CMD=0"
ALIGN_OFF="$ALIGN_OFF MICRODUCK_ABDUCTION=0 MICRODUCK_YAW_DC=0 MICRODUCK_YAW_SYMMETRY=0"
ALIGN_OFF="$ALIGN_OFF MICRODUCK_TURN_FIRST=0 MICRODUCK_SYMMETRY=0 MICRODUCK_HIP_ROLL_STD=0.05"
ALIGN_OFF="$ALIGN_OFF MICRODUCK_POSE_ZERO_CMD_PROB=0.0"

log() { printf '[%(%F %T)T] %s\n' -1 "$*" | tee -a "$STATUS" >&2; }

# Deterministic with MICRODUCK_TASK_NAMED_RUNS=1; falls back to parsing the log
# (grep -o, not rg — ripgrep is not installed here) if the naming hook degrades,
# which is the failure that silently lost a run dir in the 2026-09-16 queue.
find_run_dir() {
  local label="$1" d
  for d in "$RUN_ROOT/$label" "$RUN_ROOT/${label}"_r*; do
    [ -d "$d" ] && { printf '%s\n' "$d"; return 0; }
  done
  grep -o 'Logging experiment in directory: .*' "$OUT/${label}_train.log" 2>/dev/null |
    tail -1 | sed 's/.*directory: //'
}

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

wait_for_gpu() {
  local label="$1" min_free="$2" waited=0 free
  while :; do
    free=$(gpu_free_mib || true)
    if [ -n "$free" ] && [ "$free" -ge "$min_free" ]; then
      log "[$label] GPU ready: ${free} MiB free after ${waited}s"
      return 0
    fi
    [ $((waited % 300)) -eq 0 ] &&
      log "[$label] waiting for driver / ${min_free} MiB free (waited ${waited}s, free=${free:-?})"
    if [ "$waited" -ge "$GPU_WAIT_MAX_S" ]; then
      log "[$label] GPU wait budget exhausted"
      return 1
    fi
    sleep 30
    waited=$((waited + 30))
  done
}

# The real smoke: builds, steps, obs 61D, every reward term computes, and the
# arm's SWITCH actually took effect (the check that caught the yaw-dc /
# symmetry mix-up in the 2026-09-16 queue).
smoke_arm() {
  local label="$1" arm_env="$2" expect="$3"
  local log_file="$OUT/${label}_smoke.log"
  log "[$label] smoke: 64 envs x ${SMOKE_ITERS}"
  env $arm_env MICRODUCK_RUN_DIR_NAME="${label}_smoke" \
    .venv/bin/python -m mjlab_microduck.train_cli "$TASK" \
      --env.scene.num-envs 64 --agent.max_iterations "$SMOKE_ITERS" \
      > "$log_file" 2>&1
  local rc=$?
  if [ "$rc" -ne 0 ] || grep -q 'Traceback (most recent call' "$log_file"; then
    log "[$label] SMOKE FAILED rc=${rc}; see ${log_file}"
    return 1
  fi
  local found
  found=$(grep -c 'Episode_Reward/head_motion' "$log_file" || true)
  if [ "$expect" = "head_on" ] && [ "$found" -eq 0 ]; then
    log "[$label] head terms ABSENT from the smoke reward log -- switch did not take effect"
    return 1
  fi
  if [ "$expect" = "head_off" ] && [ "$found" -ne 0 ]; then
    log "[$label] head terms PRESENT but this arm must be the verbatim official recipe"
    return 1
  fi
  # Prove the run-directory naming hook engaged BEFORE committing 7 GPU-hours:
  # with the toggle off, two arms collide on one timestamped directory and die.
  if ! grep -q "run directory named by task: ${label}_smoke" "$log_file"; then
    log "[$label] FATAL: task-named run dirs did not engage (needs MICRODUCK_TASK_NAMED_RUNS=1)"
    return 1
  fi
  log "[$label] smoke passed (head_motion lines=${found}, expected ${expect})"
}

# Launch one arm in the background; writes the run dir into $OUT/<label>.rundir
launch_arm() {
  local label="$1" arm_env="$2"
  local train_log="$OUT/${label}_train.log"
  log "[$label] TRAIN start: ${ITERS} iters @ ${ENVS}"
  env $arm_env MICRODUCK_RUN_DIR_NAME="$label" \
    .venv/bin/python -m mjlab_microduck.train_cli "$TASK" \
      --env.scene.num-envs "$ENVS" --agent.max_iterations "$ITERS" \
      --agent.save_interval 500 \
      > "$train_log" 2>&1 &
  echo $! > "$OUT/${label}.pid"
}

export_final_onnx() {
  local label="$1" run_dir="$2" ck onnx="$OUT/${label}.onnx"
  ck=$(ls -t "$run_dir"/model_*.pt 2>/dev/null | head -1)
  [ -z "$ck" ] && { log "[$label] no checkpoint in ${run_dir}"; return 1; }
  .venv/bin/python scripts/export.py "$TASK" --checkpoint-file "$ck" \
    --onnx-file "$onnx" > "$OUT/${label}_export.log" 2>&1 || {
      log "[$label] export failed"; return 1; }
  printf '%s\n' "$onnx"
}

# 64-env protocol (AGENTS.md: never quote single-env magnitudes).
eval_arm() {
  local label="$1" onnx="$2"
  log "[$label] eval: straightness vx=0.2 (P64)"
  .venv/bin/python scripts/eval_straightness.py --onnx "$onnx" --task "$TASK" \
    --lin-vel-x 0.2 --num-envs 64 --seed 0 --seconds 20 --freeze-pose-cmd \
    --json "$OUT/${label}_walk.json" > "$OUT/${label}_walk.log" 2>&1 ||
    log "[$label] walk eval failed"
  log "[$label] eval: static vx=0.0 (P64)"
  .venv/bin/python scripts/eval_straightness.py --onnx "$onnx" --task "$TASK" \
    --lin-vel-x 0.0 --num-envs 64 --seed 0 --seconds 20 --freeze-pose-cmd \
    --json "$OUT/${label}_stand.json" > "$OUT/${label}_stand.log" 2>&1 ||
    log "[$label] stand eval failed"
  log "[$label] eval: posture/head acceptance table"
  .venv/bin/python scripts/check_posture_acceptance.py --onnx "$onnx" --keep-json \
    > "$OUT/${label}_acceptance.log" 2>&1 ||
    log "[$label] acceptance table failed (see log)"
}

log '========================================================================'
log "official-alignment queue; fingerprint=$(fingerprint)"
log "arms: align_full (official + head block), align_official (verbatim official)"
log "budget: ${ITERS} iters @ ${ENVS} each, 2 arms in parallel"
log '========================================================================'

printf 'label\ttask\titers\tenvs\trc\trun_dir\tfp_before\tfp_after\n' > "$RESULTS"
FP_BEFORE=$(fingerprint)

wait_for_gpu queue "$GPU_MIN_FREE" || exit 1

smoke_arm align_full "$ALIGN_OFF MICRODUCK_HEAD_STABILITY=1" head_on || exit 1
smoke_arm align_official "$ALIGN_OFF MICRODUCK_HEAD_STABILITY=0" head_off || exit 1

launch_arm align_full "$ALIGN_OFF MICRODUCK_HEAD_STABILITY=1"
launch_arm align_official "$ALIGN_OFF MICRODUCK_HEAD_STABILITY=0"

rc_all=0
for label in align_full align_official; do
  pid=$(cat "$OUT/${label}.pid")
  if wait "$pid"; then rc=0; else rc=$?; fi
  run_dir=$(find_run_dir "$label")
  [ "$rc" -ne 0 ] && rc_all=1
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$label" "$TASK" "$ITERS" "$ENVS" "$rc" "${run_dir:--}" \
    "$FP_BEFORE" "$(fingerprint)" >> "$RESULTS"
  log "[$label] finished rc=${rc} run=${run_dir:--}"
done

FP_AFTER=$(fingerprint)
if [ "$FP_BEFORE" != "$FP_AFTER" ]; then
  log "CONTAMINATED: actuator fingerprint changed ${FP_BEFORE} -> ${FP_AFTER}; results are NOT interpretable"
  exit 1
fi

for label in align_full align_official; do
  run_dir=$(find_run_dir "$label")
  [ -z "$run_dir" ] || [ ! -d "$run_dir" ] && { log "[$label] no run dir; skipping export"; continue; }
  onnx=$(export_final_onnx "$label" "$run_dir") || continue
  log "[$label] onnx=${onnx}"
  eval_arm "$label" "$onnx"
done

log "queue complete; rc_all=${rc_all}"
log "results: ${RESULTS}"
log "acceptance tables: ${OUT}/*_acceptance.log"
