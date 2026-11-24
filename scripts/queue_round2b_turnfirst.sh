#!/usr/bin/env bash
# =============================================================================
# Queue round 2b — the TURN-FIRST arm (2026-09-16).
#
# Why a separate launcher: `scripts/queue_round2.sh` is already running and bash
# reads a running script by byte offset, so editing it would corrupt the queue.
# This is a single-arm, self-contained copy of the same safety rails.
#
# WHAT IT TESTS. The steering defect, which round 2 does NOT address.
#   * a +-34 deg/s yaw command moves our policy's measured yaw rate by <2 %, with
#     the wrong sign at a standstill, while `track_angular_velocity` (weight 2.0,
#     params byte-identical to the official recipe) reads 0.119 of 1.0 — the
#     policy abandons ~23 % of total reward rather than turn;
#   * the official policy tracks the same command fine (+18.5 deg/s);
#   * so the turn channel is not under-priced, it is unreachable from the forward
#     shuffle basin the policy settled in.
# Mechanism (`MICRODUCK_TURN_FIRST=1`): a command-range curriculum whose first
# 1200 iterations ask for TURNING ONLY (linear range kept small but non-zero so
# the input neurons do not die), widening to the full task by iteration 2500.
# This is the repo's own rule: give a rare-but-important command region a stage
# where it is the only thing asked for.
#
# START CONDITION. It WAITS for a free GPU slot rather than competing: it polls
# until >= 4000 MiB is free AND fewer than 2 training processes are live. An OOM
# would kill a run that already has hours of training in it.
#
# Usage:
#   setsid nohup bash scripts/queue_round2b_turnfirst.sh \
#       > logs/queue_turnfirst.out 2>&1 < /dev/null &
# =============================================================================
set -u
cd "$(dirname "$0")/.." || exit 1
export UV_CACHE_DIR=/tmp/uv-cache
export WANDB_MODE=offline

NAME=turn_first
TASK=Mjlab-Velocity-Flat-MicroDuck
ITERS=${ITERS:-9000}
ENVS=${ENVS:-2048}
OUT=logs/queue_round2
STATUS=logs/queue_turnfirst_status.txt
RESULTS="$OUT/results.tsv"
PARAMS_DIR=vendor/bam/bam/params/hd1910
SMOKE_ITERS=5
BUDGET_SLACK_S=$((2 * 3600))
MAX_WAIT_S=$((14 * 3600))

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
log "queue round 2b: TURN-FIRST arm (pid $$); fingerprint $FP_BEFORE"
log "=============================================================="

# ---- wait for a slot: <2 training processes AND enough free VRAM --------
waited=0
while :; do
  n=$(pgrep -f "train_cli Mjlab" | wc -l)
  free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
  if [ "$n" -lt 2 ] && [ "${free_mib:-0}" -ge 4000 ]; then
    log "slot free: $n training procs, ${free_mib} MiB free (waited ${waited}s)"
    break
  fi
  [ $((waited % 600)) -eq 0 ] && log "waiting: $n training procs, ${free_mib} MiB free (${waited}s)"
  if [ "$waited" -ge "$MAX_WAIT_S" ]; then
    log "gave up after ${MAX_WAIT_S}s without a free slot — NOT starting"
    exit 1
  fi
  sleep 60; waited=$((waited + 60))
done

[ -f "$RESULTS" ] || printf 'name\ttask\titers\tenvs\trun_dir\trc\tfp_before\tfp_after\telapsed_s\tonnx\teval_walk02\teval_zerocmd\teval_sitstand\n' > "$RESULTS"

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

smoke_log="logs/smoke_${NAME}.log"
train_log="train_${NAME}.log"

log "-- smoke test (64 envs, ${SMOKE_ITERS} iters), env MICRODUCK_TURN_FIRST=1"
export MICRODUCK_RUN_DIR_NAME="${NAME}_smoke"
timeout -k 60 1800 env MICRODUCK_TURN_FIRST=1 \
    .venv/bin/python -m mjlab_microduck.train_cli "$TASK" \
    --env.scene.num-envs 64 --agent.max_iterations "$SMOKE_ITERS" > "$smoke_log" 2>&1
smoke_rc=$?
if [ "$smoke_rc" -ne 0 ] || grep -qE "RuntimeError|Traceback \(most recent" "$smoke_log"; then
  log "SMOKE FAILED rc=$smoke_rc — not starting the run; see $smoke_log"
  grep -E "RuntimeError|Error|Traceback" "$smoke_log" | tail -3 | sed 's/^/   /'
  exit 1
fi
iter_t=$(grep -o "Iteration time: [0-9.]*s" "$smoke_log" | tail -1 | grep -o "[0-9.]*" || true)
[ -z "${iter_t:-}" ] && iter_t=2.0
# the smoke ran WITH the other two jobs live, so do NOT multiply by concurrency
# again — that double-counts and was worth 2x on the round-2 budgets.
est_s=$(python3 -c "print(int(float('${iter_t}')*${ITERS}*1.15))")
budget_s=$((est_s + BUDGET_SLACK_S))
log "smoke ok: ${iter_t}s/iter (under load) -> est $(python3 -c "print(f'{${est_s}/3600:.1f}')")h, budget $(python3 -c "print(f'{${budget_s}/3600:.1f}')")h"

started=$(date +%s)
export MICRODUCK_RUN_DIR_NAME="$NAME"
run_budgeted "$budget_s" "$train_log" \
    env MICRODUCK_TURN_FIRST=1 \
    .venv/bin/python -m mjlab_microduck.train_cli "$TASK" \
    --env.scene.num-envs "$ENVS" --agent.max_iterations "$ITERS" \
    --agent.save_interval 1000
trc=$?
elapsed=$(( $(date +%s) - started ))

run_dir=$(grep -o "Logging experiment in directory: .*" "$train_log" | tail -1 | sed 's/.*directory: //')
[ -z "${run_dir:-}" ] && run_dir="-"
onnx=$(ls -t "$run_dir"/*.onnx 2>/dev/null | head -1); [ -z "${onnx:-}" ] && onnx="-"
fp_after=$(fingerprint)
log "done rc=${trc} in $((elapsed/3600))h$(( (elapsed%3600)/60 ))m  run_dir=${run_dir}"
[ "$fp_after" != "$FP_BEFORE" ] && log "CONTAMINATED: fingerprint $FP_BEFORE -> $fp_after"

ev_walk="-"; ev_zero="-"
if [ "$onnx" != "-" ]; then
  dyn="$OUT/${NAME}_dyn.onnx"
  .venv/bin/python scripts/make_dynamic_batch_onnx.py --onnx "$onnx" \
      --write "$dyn" --check-batch 64 > "$OUT/${NAME}_dyn.log" 2>&1 \
    && log "dynamic-batch eval copy: $dyn" || log "WARN: dynamic rewrite failed"
  ev_onnx="$onnx"; [ -f "$dyn" ] && ev_onnx="$dyn"
  .venv/bin/python scripts/eval_straightness.py --onnx "$ev_onnx" --task "$TASK" \
      --lin-vel-x 0.2 --num-envs 64 --seed 0 --seconds 20 --freeze-pose-cmd \
      --json "$OUT/${NAME}_walk02.json" > "$OUT/${NAME}_walk02.log" 2>&1 \
    && ev_walk="$OUT/${NAME}_walk02.json"
  .venv/bin/python scripts/eval_straightness.py --onnx "$ev_onnx" --task "$TASK" \
      --lin-vel-x 0.0 --num-envs 64 --seed 0 --seconds 20 --freeze-pose-cmd \
      --json "$OUT/${NAME}_zerocmd.json" > "$OUT/${NAME}_zerocmd.log" 2>&1 \
    && ev_zero="$OUT/${NAME}_zerocmd.json"
  log "eval: walk02=${ev_walk} zerocmd=${ev_zero}"
  log "STEER CHECK (the point of this arm):"
  .venv/bin/python scripts/queue_round2_report.py --steer 2>&1 | tail -8 | sed 's/^/   /'
fi

printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
  "$NAME" "$TASK" "$ITERS" "$ENVS" "$run_dir" "$trc" "$FP_BEFORE" "$fp_after" \
  "$elapsed" "$onnx" "$ev_walk" "$ev_zero" "-" >> "$RESULTS"

{
  echo "== ${NAME} =="
  echo "task: ${TASK}   iters: ${ITERS} @ ${ENVS}   env: MICRODUCK_TURN_FIRST=1"
  echo "status: rc=${trc}   elapsed: ${elapsed}s"
  echo "run_dir: ${run_dir}"
  echo "fingerprint: ${FP_BEFORE} -> ${fp_after}"
  echo "onnx: ${onnx}"
  echo "eval: walk02=${ev_walk}  zero-cmd=${ev_zero}"
} >> "$STATUS"
log "next: uv run python scripts/queue_round2_report.py --steer --sweep --runs turn_first,walk_sym,walk_yawdc"
