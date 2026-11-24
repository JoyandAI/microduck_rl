#!/usr/bin/env bash
# =============================================================================
# Queue round 2d — runs the two remaining walking arms SEQUENTIALLY.
#
# Why this replaced two independent waiters (2026-09-16). `queue_round2b` and
# `queue_round2c` each polled the same start condition
#   (fewer than 2 training processes AND >= 4 GB VRAM free)
# so the moment the sitstand run ended they would BOTH pass, and the main
# queue's own `walk_yawdc` would start in the same window: FOUR concurrent
# 2048-env runs on a 12 GB card (~10.4 GB of weights/rollouts) — an OOM there
# kills a run that already has hours of training in it, which is the most
# expensive failure mode this queue has.
#
# This launcher runs them one at a time and re-checks the slot before each, so
# the peak is (walk_sym|walk_yawdc) + 1 = 3 concurrent runs.
#
# The arms, in order:
#   1. turn_first       MICRODUCK_TURN_FIRST=1
#                       -> ask for turning ONLY for the first 1200 iters, so the
#                          "produce yaw by placing feet" skill is discovered before
#                          a forward limit cycle exists to hide inside.
#   2. turnfirst_yawdc  MICRODUCK_TURN_FIRST=1 MICRODUCK_YAW_DC=1
#                       -> the same, plus a price on the cycle-averaged net yaw
#                          rate while walking straight. This is the arm that
#                          targets the user's requirement as a CONJUNCTION:
#                          turn when commanded AND straight when not.
#
# NOTE the switch is MICRODUCK_YAW_DC (the term is a net-yaw-rate penalty).
# The MIRROR LOSS is a different switch, MICRODUCK_SYMMETRY, and is NOT on here.
#
# Usage:
#   setsid nohup bash scripts/queue_round2d_sequential.sh \
#       > logs/queue_seq.out 2>&1 < /dev/null &
# =============================================================================
set -u
cd "$(dirname "$0")/.." || exit 1
export UV_CACHE_DIR=/tmp/uv-cache
export WANDB_MODE=offline

ITERS=${ITERS:-9000}
ENVS=${ENVS:-2048}
OUT=logs/queue_round2
STATUS=logs/queue_seq_status.txt
RESULTS="$OUT/results.tsv"
PARAMS_DIR=vendor/bam/bam/params/hd1910
SMOKE_ITERS=5
BUDGET_SLACK_S=$((2 * 3600))
MAX_WAIT_S=$((18 * 3600))
# FREE memory required before starting an arm, not total.
# Measured (RTX 4080 Laptop 12 GB): velocity@2048 = 2880 MiB, sitstand@1024 =
# 4406 MiB, CUDA/display ~1000 MiB. A new velocity run takes 2880 MiB, so demand
# 4500 MiB FREE => ~1.6 GB headroom left for an eval on top.
#
# Why not 3000: with the sitstand run still alive (the 4.4 GB one) free memory is
# only ~3.5 GB, so a 3000 threshold admits a third job into a 650 MiB headroom —
# and an OOM there kills the sitstand run, i.e. the one carrying the Q2 verdict.
# 4500 waits out the sitstand run and then admits the arm with room to spare.
MIN_FREE_MIB=${MIN_FREE_MIB:-4500}

mkdir -p "$OUT" logs
log() { echo "[$(date '+%F %T')] $*"; }

fingerprint() {
  .venv/bin/python - <<'PY'
from mjlab_microduck.train_hook import actuator_params_fingerprint
print(actuator_params_fingerprint()[0])
PY
}

FP_BEFORE=$(fingerprint)
log "=============================================================="
log "queue round 2d (SEQUENTIAL): fingerprint $FP_BEFORE"
log "arms: turn_first -> turnfirst_yawdc  (one at a time)"
log "=============================================================="

[ -f "$RESULTS" ] || printf 'name\ttask\titers\tenvs\trun_dir\trc\tfp_before\tfp_after\telapsed_s\tonnx\teval_walk02\teval_zerocmd\teval_sitstand\n' > "$RESULTS"

wait_for_slot() {  # $1 = arm label
  local waited=0 n free_mib
  while :; do
    n=$(pgrep -f "train_cli Mjlab" | wc -l)
    free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    # `< 3`, not `< 2`: the arm this queue exists for (the turn-first + yaw_dc
    # combination) was otherwise blocked until walk_sym finished, purely because
    # the guard was written for the sitstand-era memory profile. The SITSTAND run
    # is the 4.4 GB one; once it exits, three 2048-env velocity runs fit. Waiting
    # 2 extra hours for a slot that measurement says is already free is not
    # caution, it is a wasted night.
    if [ "$n" -lt 3 ] && [ "${free_mib:-0}" -ge "$MIN_FREE_MIB" ]; then
      log "[$1] slot free: $n procs, ${free_mib} MiB free (waited ${waited}s)"
      return 0
    fi
    [ $((waited % 900)) -eq 0 ] && log "[$1] waiting: $n procs, ${free_mib} MiB free (${waited}s)"
    if [ "$waited" -ge "$MAX_WAIT_S" ]; then
      log "[$1] gave up after ${MAX_WAIT_S}s without a slot"
      return 1
    fi
    sleep 60; waited=$((waited + 60))
  done
}

run_budgeted() {
  local budget="$1" out="$2"; shift 2
  setsid --wait "$@" > "$out" 2>&1 &
  local pid=$! w=0
  while kill -0 "$pid" 2>/dev/null; do
    sleep 30; w=$((w + 30))
    if [ "$w" -ge "$budget" ]; then
      log "  TIMEOUT: budget ${budget}s -> SIGTERM $pid"
      kill -TERM -"$pid" 2>/dev/null; sleep 60; kill -KILL -"$pid" 2>/dev/null
      wait "$pid" 2>/dev/null; return 124
    fi
  done
  wait "$pid"
}

run_arm() {  # $1 = name, $2 = env assignments
  local name="$1" arm_env="$2"
  local smoke_log="logs/smoke_${name}.log" train_log="train_${name}.log"

  wait_for_slot "$name" || return 1

  log "-- [$name] smoke (64 envs x ${SMOKE_ITERS}) with [$arm_env]"
  export MICRODUCK_RUN_DIR_NAME="${name}_smoke"
  timeout -k 60 1800 env $arm_env \
      .venv/bin/python -m mjlab_microduck.train_cli Mjlab-Velocity-Flat-MicroDuck \
      --env.scene.num-envs 64 --agent.max_iterations "$SMOKE_ITERS" > "$smoke_log" 2>&1
  local rc=$?
  if [ "$rc" -ne 0 ] || grep -qE "RuntimeError|Traceback \(most recent" "$smoke_log"; then
    log "[$name] SMOKE FAILED rc=$rc — skipping this arm"
    grep -E "RuntimeError|Error|Traceback" "$smoke_log" | tail -3 | sed 's/^/   /'
    return 1
  fi
  grep -q "Episode_Reward/yaw_dc" "$smoke_log" && log "   ok: yaw_dc term active"
  local iter_t
  iter_t=$(grep -o "Iteration time: [0-9.]*s" "$smoke_log" | tail -1 | grep -o "[0-9.]*" || true)
  [ -z "${iter_t:-}" ] && iter_t=2.0
  # the smoke ran WITH the other jobs live -> do NOT multiply by concurrency again
  local est_s budget_s
  est_s=$(python3 -c "print(int(float('${iter_t}')*${ITERS}*1.15))")
  budget_s=$((est_s + BUDGET_SLACK_S))
  log "   smoke ok: ${iter_t}s/iter (under load) -> est $(python3 -c "print(f'{${est_s}/3600:.1f}')")h, budget $(python3 -c "print(f'{${budget_s}/3600:.1f}')")h"

  local started; started=$(date +%s)
  export MICRODUCK_RUN_DIR_NAME="$name"
  run_budgeted "$budget_s" "$train_log" \
      env $arm_env .venv/bin/python -m mjlab_microduck.train_cli \
      Mjlab-Velocity-Flat-MicroDuck \
      --env.scene.num-envs "$ENVS" --agent.max_iterations "$ITERS" \
      --agent.save_interval 1000
  local trc=$?
  local elapsed=$(( $(date +%s) - started ))

  local run_dir onnx fp_after
  run_dir=$(grep -o "Logging experiment in directory: .*" "$train_log" | tail -1 | sed 's/.*directory: //')
  [ -z "${run_dir:-}" ] && run_dir="-"
  onnx=$(ls -t "$run_dir"/*.onnx 2>/dev/null | head -1); [ -z "${onnx:-}" ] && onnx="-"
  fp_after=$(fingerprint)
  log "[$name] done rc=${trc} in $((elapsed/3600))h$(( (elapsed%3600)/60 ))m  run_dir=${run_dir}"
  [ "$fp_after" != "$FP_BEFORE" ] && log "[$name] CONTAMINATED: $FP_BEFORE -> $fp_after"

  local ev_walk="-" ev_zero="-"
  if [ "$onnx" != "-" ]; then
    local dyn="$OUT/${name}_dyn.onnx"
    .venv/bin/python scripts/make_dynamic_batch_onnx.py --onnx "$onnx" \
        --write "$dyn" --check-batch 64 > "$OUT/${name}_dyn.log" 2>&1
    local ev_onnx="$onnx"; [ -f "$dyn" ] && ev_onnx="$dyn"
    .venv/bin/python scripts/eval_straightness.py --onnx "$ev_onnx" \
        --lin-vel-x 0.2 --num-envs 64 --seed 0 --seconds 20 --freeze-pose-cmd \
        --json "$OUT/${name}_walk02.json" > "$OUT/${name}_walk02.log" 2>&1 \
      && ev_walk="$OUT/${name}_walk02.json"
    .venv/bin/python scripts/eval_straightness.py --onnx "$ev_onnx" \
        --lin-vel-x 0.0 --num-envs 64 --seed 0 --seconds 20 --freeze-pose-cmd \
        --json "$OUT/${name}_zerocmd.json" > "$OUT/${name}_zerocmd.log" 2>&1 \
      && ev_zero="$OUT/${name}_zerocmd.json"
    log "[$name] eval: walk02=${ev_walk} zerocmd=${ev_zero}"
  fi

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$name" "Mjlab-Velocity-Flat-MicroDuck" "$ITERS" "$ENVS" "$run_dir" "$trc" \
    "$FP_BEFORE" "$fp_after" "$elapsed" "$onnx" "$ev_walk" "$ev_zero" "-" >> "$RESULTS"

  {
    echo "== ${name} =="
    echo "env: $arm_env   iters: ${ITERS} @ ${ENVS}"
    echo "status: rc=${trc}   elapsed: ${elapsed}s"
    echo "run_dir: ${run_dir}"
    echo "fingerprint: ${FP_BEFORE} -> ${fp_after}"
    echo "onnx: ${onnx}"
    echo "eval: walk02=${ev_walk}  zero-cmd=${ev_zero}"
  } >> "$STATUS"
}

# ORDER MATTERS. The combined arm goes FIRST: it is the one that can satisfy the
# user's requirement as stated ("turn when commanded AND straight otherwise"), so
# knowing whether it works is worth more, sooner, than isolating the turn
# mechanism. `turn_first` alone still runs second — if the combined arm fails, its
# single-mechanism twin tells us WHICH half failed.
run_arm turnfirst_yawdc "MICRODUCK_TURN_FIRST=1 MICRODUCK_YAW_DC=1"
run_arm turn_first      "MICRODUCK_TURN_FIRST=1"

log "=============================================================="
log "sequential queue finished. results: $RESULTS"
log "judge: uv run python scripts/queue_round2_report.py --steer --sweep"
