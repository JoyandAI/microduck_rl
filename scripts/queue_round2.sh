#!/usr/bin/env bash
# =============================================================================
# Queue round 2 (2026-09-16 night) — the two MEASURED defects, one arm each.
#
# What the measurements said before this queue was written:
#
#   Q1 veer (64 env x 5 seed, scripts/q1_straightness_matrix.sh):
#       official alpha_walking   yaw DC -0.23 deg/s   turning radius ~12 m
#       our walk_hd1910_12000   yaw DC -12.08 deg/s   turning radius  3.2 m
#     The old "hip_yaw parked on the stop / feet twisted 28 deg" pathology is GONE
#     (0.0 % of steps at a stop, foot-pair torsion < 1 deg); what is left is a
#     roughly CONSTANT net yaw rate. Two mechanisms can price it, and this queue
#     tests them as separate arms so the result is attributable:
#       W1 MICRODUCK_SYMMETRY=1      -> PPO mirror loss (tasks/symmetry.py 61-D table,
#                                       now proven correct by tests/test_symmetry_table.py)
#       W2 MICRODUCK_YAW_SYMMETRY=1  -> reward term on EMA(wz) while walking straight
#                                       (tests/test_yaw_symmetry_term.py)
#     Both arms inherit everything else (heading_hold, yaw_hold_zero_cmd, the
#     stillness trio), so the arms isolate ONE mechanism each.
#
#   Q2 head droop (scripts/sitstand_com_support_check.py + policy_failure_analysis §8):
#     the commanded stand pose is statically fine -- CoM 0.41 mm from the ankle
#     axis and +17.1 mm inside the support polygon -- and folding the neck to its
#     hard stop moves the CoM 29.8 mm FORWARD and shrinks the margin to 0.96 mm.
#     So it is not a centre-of-mass repair; the reward paid for the head-tripod
#     rest (+3.69/step over standing upright). S1 trains the fix: the head-ground
#     LOAD is charged once the posture ramp has settled.
#
# Safety rails (all inherited from the 09-10 accident post-mortem):
#   * actuator-parameter fingerprint taken before and re-verified after; a run
#     whose digest changed is marked CONTAMINATED and must not be interpreted.
#   * smoke test (64 envs, 5 iters) first — it is also what catches an rsl_rl
#     mirror-loss wiring error, which has never been exercised in a real run.
#   * budget = smoke_iter_time x iters x 1.15 x CONCURRENCY + 2 h (an idle-GPU
#     estimate is wrong the moment two jobs share the card).
#
# Usage:
#   setsid nohup bash scripts/queue_round2.sh > logs/queue_round2.out 2>&1 < /dev/null &
#   cat logs/queue_round2_status.txt ; tail -f logs/queue_round2.out
#   ONLY=walk_yawdc bash scripts/queue_round2.sh          # one arm
# =============================================================================
set -u
cd "$(dirname "$0")/.." || exit 1

export WANDB_MODE=offline
export UV_CACHE_DIR=/tmp/uv-cache

OUT=logs/queue_round2
STATUS=logs/queue_round2_status.txt
RESULTS="$OUT/results.tsv"
PARAMS_DIR=vendor/bam/bam/params/hd1910
SMOKE_ITERS=5
BUDGET_SLACK_S=$((2 * 3600))
GPU_WAIT_MAX_S=$((6 * 3600))
CONCURRENCY=${CONCURRENCY:-2}
ONLY="${ONLY:-}"

mkdir -p "$OUT" logs
log() { echo "[$(date '+%F %T')] $*"; }

fingerprint() {
  .venv/bin/python - <<'PY'
from mjlab_microduck.train_hook import actuator_params_fingerprint
print(actuator_params_fingerprint()[0])
PY
}

# name|task|iters|envs|extra env (space separated)|rationale
QUEUE=(
  "walk_sym|Mjlab-Velocity-Flat-MicroDuck|9000|2048|MICRODUCK_SYMMETRY=1|W1: PPO mirror loss (never exercised before; the 61-D table is now proven)"
  "sitstand_headfix|Mjlab-SitStand-Flat-MicroDuck|8000|1024||S1: charge the head-tripod LOAD once the posture ramp settles"
  "walk_yawdc|Mjlab-Velocity-Flat-MicroDuck|9000|2048|MICRODUCK_YAW_SYMMETRY=1|W2: dense reward on EMA(wz) while walking straight"
)

FP_BEFORE=$(fingerprint)
STAMP=$(date '+%Y%m%d_%H%M%S')
SNAP="$OUT/params_snapshot_$STAMP"
mkdir -p "$SNAP"; cp -a "$PARAMS_DIR"/. "$SNAP"/

log "=============================================================="
log "queue round 2 (pid $$); actuator fingerprint $FP_BEFORE"
log "params snapshot $SNAP"
log "concurrency $CONCURRENCY"
log "arms: W1 walk_sym | W2 walk_yawdc | S1 sitstand_headfix"
log "=============================================================="

waited=0
while pgrep -f "train_cli Mjlab|train Mjlab" > /dev/null 2>&1; do
  [ $((waited % 600)) -eq 0 ] && log "waiting: another training is running (${waited}s)"
  [ "$waited" -ge "$GPU_WAIT_MAX_S" ] && { log "WARNING: GPU busy ${GPU_WAIT_MAX_S}s — proceeding"; break; }
  sleep 60; waited=$((waited + 60))
done
[ "$waited" -gt 0 ] && log "GPU free (waited ${waited}s)"

[ -f "$RESULTS" ] || printf 'name\ttask\titers\tenvs\trun_dir\trc\tfp_before\tfp_after\telapsed_s\tonnx\teval_walk02\teval_zerocmd\teval_sitstand\n' > "$RESULTS"

run_budgeted() {  # $1=budget seconds, $2=log, $3..=command
  local budget="$1" out="$2"; shift 2
  setsid --wait "$@" > "$out" 2>&1 &
  local pid=$! waited=0
  while kill -0 "$pid" 2>/dev/null; do
    sleep 30; waited=$((waited + 30))
    if [ "$waited" -ge "$budget" ]; then
      log "  TIMEOUT: budget ${budget}s -> SIGTERM process group $pid"
      kill -TERM -"$pid" 2>/dev/null; sleep 60; kill -KILL -"$pid" 2>/dev/null
      wait "$pid" 2>/dev/null; return 124
    fi
  done
  wait "$pid"
}

run_one() {
  local name="$1" task="$2" iters="$3" envs="$4" extra="$5" why="$6"
  local smoke_log="logs/smoke_${name}.log" train_log="train_${name}.log"
  local started; started=$(date +%s)
  local run_dir="-" onnx="-" trc=0 elapsed=0
  local ev_walk="-" ev_zero="-" ev_sit="-"

  log "-- RUN ${name}: ${task} | ${iters} iters @ ${envs} envs | env: ${extra:-（默认）}"
  log "   why: ${why}"

  # NB: `VAR=x some_function` does NOT reliably export VAR to the children a bash
  # function spawns — the assignment is applied to the shell, not marked for
  # export. Export explicitly, or the run lands in a timestamped directory and the
  # checkpoint sweep can never find it.
  export MICRODUCK_RUN_DIR_NAME="${name}_smoke"
  timeout -k 60 1800 \
      env $extra .venv/bin/python -m mjlab_microduck.train_cli "$task" \
      --env.scene.num-envs "$envs" --agent.max_iterations "$SMOKE_ITERS" \
      > "$smoke_log" 2>&1
  local smoke_rc=$?
  if [ "$smoke_rc" -ne 0 ] || grep -qE "RuntimeError|Traceback \(most recent" "$smoke_log"; then
    log "   SMOKE FAILED (rc=$smoke_rc) — skipping ${name}; see ${smoke_log}"
    grep -E "RuntimeError|Error|Traceback" "$smoke_log" | tail -3 | sed 's/^/     /'
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$name" "$task" "$iters" "$envs" "smoke-failed" "$smoke_rc" "-" "-" 0 "-" "-" "-" "-" >> "$RESULTS"
    return 1
  fi
  local iter_t
  iter_t=$(grep -o "Iteration time: [0-9.]*s" "$smoke_log" | tail -1 | grep -o "[0-9.]*" || true)
  [ -z "${iter_t:-}" ] && { iter_t=2.0; log "   WARNING: no Iteration time in smoke log; assuming ${iter_t}s/iter"; }
  CONC_PENALTY=${CONC_PENALTY:-$CONCURRENCY}
  local est_s budget_s
  est_s=$(python3 -c "print(int(float('${iter_t}')*${iters}*1.15*${CONC_PENALTY}))")
  budget_s=$((est_s + BUDGET_SLACK_S))
  log "   smoke ok: ${iter_t}s/iter (alone) -> est $(python3 -c "print(f'{${est_s}/3600:.1f}')")h, budget $(python3 -c "print(f'{${budget_s}/3600:.1f}')")h"

  export MICRODUCK_RUN_DIR_NAME="$name"
  run_budgeted "$budget_s" "$train_log" \
      env $extra .venv/bin/python -m mjlab_microduck.train_cli "$task" \
      --env.scene.num-envs "$envs" --agent.max_iterations "$iters" \
      --agent.save_interval 1000
  trc=$?
  elapsed=$(( $(date +%s) - started ))

  run_dir=$(grep -o "Logging experiment in directory: .*" "$train_log" | tail -1 | sed 's/.*directory: //')
  [ -z "${run_dir:-}" ] && run_dir="-"
  onnx=$(ls -t "$run_dir"/*.onnx 2>/dev/null | head -1); [ -z "${onnx:-}" ] && onnx="-"
  log "   done rc=${trc} in $((elapsed/3600))h$(( (elapsed%3600)/60 ))m"
  log "   run_dir=${run_dir}  onnx=${onnx}"

  local fp_after; fp_after=$(fingerprint)
  if [ "$fp_after" != "$FP_BEFORE" ]; then
    log "   CONTAMINATED: actuator fingerprint changed $FP_BEFORE -> $fp_after (discard this run)"
  fi

  if [ "$onnx" != "-" ]; then
    # Batched inference needs a symbolic batch axis, and the rewrite is PROVEN
    # bit-identical before use. Without it the evaluator does one ORT call per env
    # per step (~50x slower), which is why straightness matrices used to be
    # single-env and therefore unrepeatable.
    local dyn="$OUT/${name}_dyn.onnx"
    .venv/bin/python scripts/make_dynamic_batch_onnx.py --onnx "$onnx" \
        --write "$dyn" --check-batch 64 > "$OUT/${name}_dyn.log" 2>&1 \
      && log "   dynamic-batch eval copy: $dyn" || log "   WARN: dynamic-batch rewrite failed; using the original"

    local ev_onnx="$onnx"; [ -f "$dyn" ] && ev_onnx="$dyn"

    if [ "$task" = "Mjlab-Velocity-Flat-MicroDuck" ]; then
      # The metric that answers the user's question: net yaw rate, turning radius,
      # path efficiency against the documented definition (along / TOTAL path).
      timeout 1800 .venv/bin/python scripts/eval_straightness.py \
        --onnx "$ev_onnx" --task "$task" --lin-vel-x 0.2 --num-envs 64 --seed 0 \
        --seconds 20 --freeze-pose-cmd --json "$OUT/${name}_walk02.json" \
        > "$OUT/${name}_walk02.log" 2>&1 && ev_walk="$OUT/${name}_walk02.json"
      timeout 1800 .venv/bin/python scripts/eval_straightness.py \
        --onnx "$ev_onnx" --task "$task" --lin-vel-x 0.0 --num-envs 64 --seed 0 \
        --seconds 20 --freeze-pose-cmd --json "$OUT/${name}_zerocmd.json" \
        > "$OUT/${name}_zerocmd.log" 2>&1 && ev_zero="$OUT/${name}_zerocmd.json"
    else
      timeout 2400 .venv/bin/python scripts/eval_sitstand_head.py \
        --onnx "$ev_onnx" --task "$task" --episode-length-s 60 \
        --params-json "$PARAMS_DIR/m5.json" > "$OUT/${name}_eval.log" 2>&1 \
        && ev_sit="$OUT/${name}_eval.log"
    fi
    log "   eval: walk02=${ev_walk} zerocmd=${ev_zero} sitstand=${ev_sit}"
  else
    log "   WARN: no onnx, skipping evaluation"
  fi

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$name" "$task" "$iters" "$envs" "$run_dir" "$trc" "$FP_BEFORE" "$fp_after" \
    "$elapsed" "$onnx" "$ev_walk" "$ev_zero" "$ev_sit" >> "$RESULTS"

  {
    echo "== ${name} =="
    echo "task: ${task}   iters: ${iters} @ ${envs}   env: ${extra:-（default）}"
    echo "status: rc=${trc} $([ "$trc" -eq 124 ] && echo '(killed over budget)' || echo '(clean)')"
    echo "elapsed: ${elapsed}s ($(python3 -c "print(f'{${elapsed}/3600:.2f}')")h)"
    echo "run_dir: ${run_dir}"
    echo "fingerprint: ${FP_BEFORE} -> ${fp_after} $([ "$fp_after" != "$FP_BEFORE" ] && echo 'CHANGED!')"
    echo "onnx: ${onnx}"
    echo "eval: walk0.2=${ev_walk}  zero-cmd=${ev_zero}  sitstand=${ev_sit}"
    echo
  } >> "$STATUS"
}

pids=()
for entry in "${QUEUE[@]}"; do
  IFS='|' read -r name task iters envs extra why <<< "$entry"
  if [ -n "$ONLY" ] && [[ ",${ONLY}," != *",${name},"* ]]; then continue; fi
  while [ "$(jobs -rp | wc -l)" -ge "$CONCURRENCY" ]; do sleep 20; done
  free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
  while [ "${free_mib:-0}" -lt 3500 ]; do
    log "waiting for VRAM (free=${free_mib} MiB)"; sleep 60
    free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
  done
  run_one "$name" "$task" "$iters" "$envs" "$extra" "$why" &
  pids+=($!)
  sleep 90
done
wait "${pids[@]}" 2>/dev/null || true

log "=============================================================="
log "queue finished. results: $RESULTS"
log "fingerprint $FP_BEFORE (any run with a different fp_after is void)"
{
  echo "queue finished: $(date '+%F %T')"
  echo "results: ${RESULTS}"
  echo "params snapshot: ${SNAP}   fingerprint: ${FP_BEFORE}"
} >> "$STATUS"
log "next: uv run python scripts/queue_round2_report.py   # verdict table"
