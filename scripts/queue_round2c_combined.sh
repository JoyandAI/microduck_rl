#!/usr/bin/env bash
# =============================================================================
# Queue round 2c — the COMBINED arm: turn-first + net-yaw-rate pricing.
#
# The user's requirement, verbatim (2026-09-16):
#   "我是要求在给出左右指令的时候才会左右转，其余时候都是直走，
#    也不是完全剥夺了转向能力"
# i.e. a CONJUNCTION of two properties, not a trade-off:
#   (1) with no turn command  -> walk straight (no drift, no circling);
#   (2) with a turn command   -> actually turn, tracking the command.
# A policy that walks straight because it CANNOT turn does not satisfy this.
#
# Why this arm is needed. The arms already in flight each cover ONE half:
#   walk_yawdc  prices |EMA(wz)| but its gate only opens for STRAIGHT commands,
#               so it fixes (1) and says nothing about (2);
#   walk_sym    makes the policy mirror-equivariant; a mirror-symmetric policy on
#               a per-env-asymmetric realization can still circle, and symmetry
#               does not create a turn channel either;
#   turn_first  creates (2) by asking for turning only, before a forward limit
#               cycle exists — but nothing in it prevents the drift returning.
# The deliverable is the arm that has BOTH switches on.
#
# Current best known (for the acceptance table):
#   delivered walk_hd1910_12000 : yawDC -12.18 deg/s, steer response -2.93 deg/s
#                                 -> fails BOTH halves
#   ck7000 (checkpoint pick)    : yawDC  -0.62 deg/s, steer response  <2 %
#                                 -> satisfies (1), FAILS (2):
#                                    straight but unsteerable, NOT shippable
#                                    under the user's criterion
#   official alpha_walking      : yawDC  -0.58 deg/s, steer response +18.5 deg/s
#                                 -> satisfies both; this is the bar to match
#
# Usage:
#   setsid nohup bash scripts/queue_round2c_combined.sh \
#       > logs/queue_combined.out 2>&1 < /dev/null &
# =============================================================================
set -u
cd "$(dirname "$0")/.." || exit 1
export UV_CACHE_DIR=/tmp/uv-cache
export WANDB_MODE=offline

NAME=${NAME:-turnfirst_yawdc}
ITERS=${ITERS:-9000}
ENVS=${ENVS:-2048}
# both halves of the requirement, one switch each:
#   MICRODUCK_TURN_FIRST=1  -> ask for turning only, first   (creates (2))
#   MICRODUCK_YAW_DC=1      -> price |EMA(wz)| on straight cmds (creates (1))
# NOTE the switch is MICRODUCK_YAW_DC (the term is a net-yaw-rate penalty).
# The MIRROR LOSS is a different switch, MICRODUCK_SYMMETRY, and is NOT on here.
ARM_ENV="MICRODUCK_TURN_FIRST=1 MICRODUCK_YAW_DC=1"

OUT=logs/queue_round2
STATUS=logs/queue_combined_status.txt
RESULTS="$OUT/results.tsv"
PARAMS_DIR=vendor/bam/bam/params/hd1910
SMOKE_ITERS=5
BUDGET_SLACK_S=$((2 * 3600))
MAX_WAIT_S=$((16 * 3600))

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
log "queue round 2c: COMBINED arm ($NAME) env: $ARM_ENV"
log "fingerprint $FP_BEFORE"
log "requirement: turn on command AND straight without one (conjunction)"
log "=============================================================="

waited=0
while :; do
  n=$(pgrep -f "train_cli Mjlab" | wc -l)
  free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
  if [ "$n" -lt 2 ] && [ "${free_mib:-0}" -ge 4000 ]; then
    log "slot free: $n training procs, ${free_mib} MiB free (waited ${waited}s)"
    break
  fi
  [ $((waited % 900)) -eq 0 ] && log "waiting: $n procs, ${free_mib} MiB free (${waited}s)"
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

log "-- smoke (64 envs x ${SMOKE_ITERS} iters) with [$ARM_ENV]"
export MICRODUCK_RUN_DIR_NAME="${NAME}_smoke"
timeout -k 60 1800 env $ARM_ENV \
    .venv/bin/python -m mjlab_microduck.train_cli Mjlab-Velocity-Flat-MicroDuck \
    --env.scene.num-envs 64 --agent.max_iterations "$SMOKE_ITERS" > "$smoke_log" 2>&1
smoke_rc=$?
if [ "$smoke_rc" -ne 0 ] || grep -qE "RuntimeError|Traceback \(most recent" "$smoke_log"; then
  log "SMOKE FAILED rc=$smoke_rc — not starting; see $smoke_log"
  grep -E "RuntimeError|Error|Traceback" "$smoke_log" | tail -3 | sed 's/^/   /'
  exit 1
fi
# sanity: BOTH mechanisms must be visibly present in the smoke log
grep -q "turn_first" "$smoke_log" && log "   ok: turn_first curriculum active"
grep -q "Episode_Reward/yaw_dc" "$smoke_log" && log "   ok: yaw_dc term active" \
    || log "   WARN: no yaw_dc term in the smoke log"

iter_t=$(grep -o "Iteration time: [0-9.]*s" "$smoke_log" | tail -1 | grep -o "[0-9.]*" || true)
[ -z "${iter_t:-}" ] && iter_t=2.0
est_s=$(python3 -c "print(int(float('${iter_t}')*${ITERS}*1.15))")
budget_s=$((est_s + BUDGET_SLACK_S))
log "smoke ok: ${iter_t}s/iter -> est $(python3 -c "print(f'{${est_s}/3600:.1f}')")h, budget $(python3 -c "print(f'{${budget_s}/3600:.1f}')")h"

started=$(date +%s)
export MICRODUCK_RUN_DIR_NAME="$NAME"
run_budgeted "$budget_s" "$train_log" \
    env $ARM_ENV .venv/bin/python -m mjlab_microduck.train_cli \
    Mjlab-Velocity-Flat-MicroDuck \
    --env.scene.num-envs "$ENVS" --agent.max_iterations "$ITERS" \
    --agent.save_interval 1000
trc=$?
elapsed=$(( $(date +%s) - started ))

run_dir=$(grep -o "Logging experiment in directory: .*" "$train_log" | tail -1 | sed 's/.*directory: //')
[ -z "${run_dir:-}" ] && run_dir="-"
onnx=$(ls -t "$run_dir"/*.onnx 2>/dev/null | head -1); [ -z "${onnx:-}" ] && onnx="-"
fp_after=$(fingerprint)
log "done rc=${trc} in $((elapsed/3600))h$(( (elapsed%3600)/60 ))m  run_dir=${run_dir}"
[ "$fp_after" != "$FP_BEFORE" ] && log "CONTAMINATED: $FP_BEFORE -> $fp_after"

ev_walk="-"; ev_zero="-"
if [ "$onnx" != "-" ]; then
  dyn="$OUT/${NAME}_dyn.onnx"
  .venv/bin/python scripts/make_dynamic_batch_onnx.py --onnx "$onnx" \
      --write "$dyn" --check-batch 64 > "$OUT/${NAME}_dyn.log" 2>&1
  ev_onnx="$onnx"; [ -f "$dyn" ] && ev_onnx="$dyn"
  .venv/bin/python scripts/eval_straightness.py --onnx "$ev_onnx" \
      --lin-vel-x 0.2 --num-envs 64 --seed 0 --seconds 20 --freeze-pose-cmd \
      --json "$OUT/${NAME}_walk02.json" > "$OUT/${NAME}_walk02.log" 2>&1 \
    && ev_walk="$OUT/${NAME}_walk02.json"
  .venv/bin/python scripts/eval_straightness.py --onnx "$ev_onnx" \
      --lin-vel-x 0.0 --num-envs 64 --seed 0 --seconds 20 --freeze-pose-cmd \
      --json "$OUT/${NAME}_zerocmd.json" > "$OUT/${NAME}_zerocmd.log" 2>&1 \
    && ev_zero="$OUT/${NAME}_zerocmd.json"
  log "eval: walk02=${ev_walk} zerocmd=${ev_zero}"
fi

printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
  "$NAME" "Mjlab-Velocity-Flat-MicroDuck" "$ITERS" "$ENVS" "$run_dir" "$trc" \
  "$FP_BEFORE" "$fp_after" "$elapsed" "$onnx" "$ev_walk" "$ev_zero" "-" >> "$RESULTS"

{
  echo "== ${NAME} =="
  echo "env: $ARM_ENV   iters: ${ITERS} @ ${ENVS}"
  echo "status: rc=${trc}   elapsed: ${elapsed}s"
  echo "run_dir: ${run_dir}"
  echo "fingerprint: ${FP_BEFORE} -> ${fp_after}"
  echo "onnx: ${onnx}"
  echo "eval: walk02=${ev_walk}  zero-cmd=${ev_zero}"
} >> "$STATUS"
log "judge with: uv run python scripts/queue_round2_report.py --steer --sweep --runs $NAME"
