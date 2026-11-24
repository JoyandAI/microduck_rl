#!/usr/bin/env bash
# Focused recovery queue for the two unresolved HD1910 policy defects:
#
#   W4: walking hip-roll splay.  R3 showed that hip-roll L1 is the only
#       promising lever (0.454 -> 0.221 rad), but its 0.10 rad deadband left
#       the residual visible.  This arm changes ONLY that hip-roll deadband.
#   S4: sit-stand folded neck.  R3's simultaneous sit/stand + head-L1 run
#       chose "never sit".  This queue first learns an upright STAND endpoint,
#       then resumes the SAME policy through low-SIT and full-SIT command mixes.
#
# It is deliberately serial.  A velocity@2048 and sitstand@1024 fit together,
# but measured contention roughly doubles velocity iteration time and makes an
# interrupted long run more likely.  Every phase starts with a 64x5 smoke test
# and records the HD1910 parameter fingerprint.

set -u
cd "$(dirname "$0")/.." || exit 1

export UV_CACHE_DIR=/tmp/uv-cache
export MPLCONFIGDIR=/tmp/matplotlib
export WANDB_MODE=offline

OUT=logs/queue_round4
STATUS="$OUT/status.txt"
RESULTS="$OUT/results.tsv"
GPU_WAIT_MAX_S=${GPU_WAIT_MAX_S:-86400}
SMOKE_ITERS=5

mkdir -p "$OUT"

# Log to the status file and stderr.  `train_phase` returns its run directory
# via stdout, so putting status lines on stdout would corrupt command
# substitutions such as `W4_DIR=$(train_phase ...)`.
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

wait_for_gpu() {
  local label="$1" min_free="$2" waited=0 free
  while :; do
    free=$(gpu_free_mib || true)
    if [ -n "$free" ] && [ "$free" -ge "$min_free" ]; then
      log "[$label] GPU ready: ${free} MiB free after ${waited}s"
      return 0
    fi
    if [ $((waited % 600)) -eq 0 ]; then
      log "[$label] waiting for a usable GPU driver / ${min_free} MiB free (waited ${waited}s)"
    fi
    if [ "$waited" -ge "$GPU_WAIT_MAX_S" ]; then
      log "[$label] GPU wait budget exhausted after ${GPU_WAIT_MAX_S}s"
      return 1
    fi
    sleep 60
    waited=$((waited + 60))
  done
}

latest_checkpoint() {
  local run_dir="$1"
  ls -t "$run_dir"/model_*.pt 2>/dev/null | head -1
}

smoke() {
  local label="$1" task="$2" arm_env="$3" log_file="$OUT/${label}_smoke.log"
  log "[$label] smoke: ${task}, 64 envs x ${SMOKE_ITERS}, ${arm_env}"
  env $arm_env MICRODUCK_RUN_DIR_NAME="${label}_smoke" \
    .venv/bin/python -m mjlab_microduck.train_cli "$task" \
      --env.scene.num-envs 64 --agent.max_iterations "$SMOKE_ITERS" \
      > "$log_file" 2>&1
  local rc=$?
  # `nan_state` always appears in the manager's startup table, including a
  # healthy run.  Only a traceback/runtime failure is fatal here; the per-step
  # Episode_Termination/nan_state metric remains available for supervision.
  if [ "$rc" -ne 0 ] || rg -q 'Traceback \(most recent|RuntimeError' "$log_file"; then
    log "[$label] SMOKE FAILED (rc=${rc}); see ${log_file}"
    return 1
  fi
  log "[$label] smoke passed"
}

train_phase() {
  local label="$1" task="$2" envs="$3" iters="$4" arm_env="$5" resume_ck="${6:-}"
  local train_log="$OUT/${label}_train.log" run_dir
  wait_for_gpu "$label" 6000 || return 1
  smoke "$label" "$task" "$arm_env" || return 1
  local fp_before
  fp_before=$(fingerprint)
  log "[$label] start: ${iters} iterations @ ${envs}; fingerprint=${fp_before}"
  if [ -n "$resume_ck" ]; then
    env $arm_env MICRODUCK_RUN_DIR_NAME="$label" \
      .venv/bin/python -m mjlab_microduck.train_cli "$task" \
        --env.scene.num-envs "$envs" --agent.max_iterations "$iters" \
        --agent.save_interval 500 --agent.load-checkpoint "$resume_ck" --agent.resume True \
        > "$train_log" 2>&1
  else
    env $arm_env MICRODUCK_RUN_DIR_NAME="$label" \
      .venv/bin/python -m mjlab_microduck.train_cli "$task" \
        --env.scene.num-envs "$envs" --agent.max_iterations "$iters" \
        --agent.save_interval 500 > "$train_log" 2>&1
  fi
  local rc=$?
  run_dir=$(rg -o 'Logging experiment in directory: .*' "$train_log" | tail -1 | sed 's/.*directory: //')
  local fp_after
  fp_after=$(fingerprint)
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$label" "$task" "$iters" "$envs" "$rc" "${run_dir:--}" "$fp_before" "$fp_after" >> "$RESULTS"
  if [ "$fp_before" != "$fp_after" ]; then
    log "[$label] CONTAMINATED actuator fingerprint: ${fp_before} -> ${fp_after}"
    return 1
  fi
  if [ "$rc" -ne 0 ] || [ -z "$run_dir" ]; then
    log "[$label] TRAIN FAILED (rc=${rc}); see ${train_log}"
    return 1
  fi
  log "[$label] training complete: ${run_dir}"
  printf '%s\n' "$run_dir"
}

export_final_onnx() {
  local label="$1" task="$2" run_dir="$3" ck onnx="$OUT/${label}.onnx"
  ck=$(latest_checkpoint "$run_dir")
  if [ -z "$ck" ]; then
    log "[$label] no checkpoint found in ${run_dir}"
    return 1
  fi
  .venv/bin/python scripts/export.py "$task" --checkpoint-file "$ck" \
    --onnx-file "$onnx" > "$OUT/${label}_export.log" 2>&1 || return 1
  [ -f "$onnx" ] || return 1
  printf '%s\n' "$onnx"
}

FP=$(fingerprint)
printf 'label\ttask\titers\tenvs\trc\trun_dir\tfp_before\tfp_after\n' > "$RESULTS"
log '========================================================================'
log "round4 recovery queue started; HD1910 fingerprint=${FP}"
log 'W4: hip-roll L1 with 0.03-rad deadband; S4: stand -> low SIT -> full SIT'
log '========================================================================'

# W4 — one variable relative to the partly-successful R3 L1 arm.
W4_ENV='MICRODUCK_STANCE_HIP_ROLL=1 MICRODUCK_STANCE_HIP_ROLL_DEADBAND=0.03'
W4_DIR=$(train_phase w4_splay_db03 Mjlab-Velocity-Flat-MicroDuck 2048 9000 "$W4_ENV") || exit 1
W4_ONNX=$(export_final_onnx w4_splay_db03 Mjlab-Velocity-Flat-MicroDuck "$W4_DIR") || exit 1
for seed in 0 1 2 3 4 5 6 7; do
  .venv/bin/python scripts/compare_walk_angles.py \
    --policy "w4_splay_db03:${W4_ONNX}" --vx 0.2 --seed "$seed" \
    > "$OUT/w4_splay_seed${seed}.log" 2>&1 || log "[w4] splay eval seed ${seed} failed"
done
.venv/bin/python scripts/eval_straightness.py --onnx "$W4_ONNX" \
  --task Mjlab-Velocity-Flat-MicroDuck --lin-vel-x 0.2 --num-envs 64 \
  --seed 0 --seconds 20 --freeze-pose-cmd --json "$OUT/w4_walk.json" \
  > "$OUT/w4_walk.log" 2>&1 || log '[w4] straightness eval failed'

# S4 — endpoint-first curriculum.  Every phase uses the same observation and
# reward contract; only the command and reset distributions change.
S4_COMMON='MICRODUCK_HEAD_L1=1'
S4_A_ENV="${S4_COMMON} MICRODUCK_SIT_PROB=0 MICRODUCK_SITTING_INIT_PROB=0"
S4_A_DIR=$(train_phase s4_stand_endpoint Mjlab-SitStand-Flat-MicroDuck 1024 1500 "$S4_A_ENV") || exit 1
S4_A_CK=$(latest_checkpoint "$S4_A_DIR")
[ -n "$S4_A_CK" ] || { log '[s4] no phase-A checkpoint'; exit 1; }

S4_B_ENV="${S4_COMMON} MICRODUCK_SIT_PROB=0.25 MICRODUCK_SITTING_INIT_PROB=0.25"
S4_B_DIR=$(train_phase s4_low_sit Mjlab-SitStand-Flat-MicroDuck 1024 2500 "$S4_B_ENV" "$S4_A_CK") || exit 1
S4_B_CK=$(latest_checkpoint "$S4_B_DIR")
[ -n "$S4_B_CK" ] || { log '[s4] no phase-B checkpoint'; exit 1; }

S4_C_ENV="${S4_COMMON} MICRODUCK_SIT_PROB=0.5 MICRODUCK_SITTING_INIT_PROB=0.5"
S4_C_DIR=$(train_phase s4_full_sitstand Mjlab-SitStand-Flat-MicroDuck 1024 5000 "$S4_C_ENV" "$S4_B_CK") || exit 1
S4_ONNX=$(export_final_onnx s4_full_sitstand Mjlab-SitStand-Flat-MicroDuck "$S4_C_DIR") || exit 1

OUT="$OUT/s4_seed_sweep" ARMS="${S4_ONNX}:s4_full_sitstand" \
  SEEDS='0 1 2 3 4 5 6 7' MIN_FREE_MIB=1200 \
  bash scripts/sit_seed_sweep.sh > "$OUT/s4_seed_sweep.log" 2>&1 || \
  log '[s4] seeded sit-stand evaluation failed'

log '========================================================================'
log "round4 complete. Results: ${RESULTS}"
log "W4 seed logs: ${OUT}/w4_splay_seed*.log"
log "S4 seed summary: ${OUT}/s4_seed_sweep/summary.tsv"
log '========================================================================'
