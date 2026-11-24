#!/usr/bin/env bash
# =============================================================================
# Queue arm S2 — sit-stand with the UNDILUTED head L1 (`MICRODUCK_HEAD_L1=1`).
#
# WHY THIS ARM (2026-09-16 evening)
#
# S1 (`sitstand_headfix`, the gated head-ground charge + neck limit proximity)
# fixed the user's reported defect -- the head bearing the robot -- and did it
# robustly: head-ground force 2.12-3.73 N -> 0.000 N in 32/32 seeded draws
# (same-physics pre-fix control: 24/24 draws loaded). But it ALSO changed the sit
# half: 20/32 draws reached the seated height inside the 4 s window vs 24/24 for
# the pre-fix control at the same training stage (Fisher p = 0.019), and the slow
# draws hold a 20 deg forward lean at UNCHANGED height for 5-10 s, two of them
# never seating inside 12 s. Exporting the PRE-fix run's own 4000/6000
# checkpoints and running the same 12 seeds is what makes this attributable to the
# change rather than to mid-training.
#
# WHAT THIS ARM IS **NOT** CLAIMING. An earlier draft of this header asserted that
# the hesitant descent was unpriced because `posture_composite` is multiplicative
# and its head factor has collapsed to ~0.002, killing the height factor's
# gradient. THAT IS WRONG and was retracted after reading the reward stack: the
# descent is priced by THREE height terms that are NOT multiplicative --
# `posture_height` (w 1.0, std 0.04), `posture_height_sharp` (w 1.0, std 0.015)
# and `posture_height_l1` (w 6.0) -- so a collapsed head factor does not reach
# them. The arithmetic also fails: the travel is only ~41 mm (stand 0.101 -> sit
# 0.060), which at the 0.05 m/s cap of `descent_speed` takes 0.82 s, so the speed
# cap cannot explain a 5-10 s hover either. **The mechanism of the hesitating sit
# is OPEN.** This arm does not test it.
#
# So this arm targets the OTHER documented residual: the neck parked on its hard
# stop (`neck_rel` = -1.934 in every phase, 32/32 draws). The undiluted head L1 is
# implemented, unit-tested and CALIBRATED at the real operating point (weighted
# -1.84/step = 22.7 % of total reward, gradient -1.0/rad against the composite's
# -0.09/rad), and it has never been trained.
#
# What this arm must show:
#   1. the neck UNFOLDS (neck_rel leaves -1.934),                        [target]
#   2. the head-ground force STAYS at 0 N,                                [no regression]
#   3. walk/sit/rise all still work, and the SIT is scored on >= 8 seeds. [no regression]
# It is NOT expected to repair the hesitating sit; if it happens to, that is a
# bonus and must still be shown on a distribution, not on one draw.
#
# Safety rails, same as every other arm here: actuator fingerprint before/after,
# 64-env smoke test first, budget from the SMOKE's own iteration time measured
# UNDER LOAD, and a VRAM gate rather than a process-count gate.
#
# Usage:
#   setsid nohup bash scripts/queue_sit_head_l1.sh > logs/queue_sit_l1.out 2>&1 < /dev/null &
# =============================================================================
set -u
cd "$(dirname "$0")/.." || exit 1
export UV_CACHE_DIR=/tmp/uv-cache
export WANDB_MODE=offline

NAME=${NAME:-sit_head_l1}
ITERS=${ITERS:-8000}
ENVS=${ENVS:-1024}
SEEDS=${SEEDS:-"0 1 2 3 4 5 6 7"}
ARM_ENV="MICRODUCK_HEAD_L1=1"

OUT=logs/queue_round2
EVALDIR=logs/sit_l1_eval
STATUS=logs/queue_sit_l1_status.txt
RESULTS="$OUT/results.tsv"
PARAMS_DIR=vendor/bam/bam/params/hd1910
SMOKE_ITERS=5
BUDGET_SLACK_S=$((2 * 3600))
MAX_WAIT_S=$((18 * 3600))
# sitstand@1024 measured at 4406 MiB (the BIG one -- heavier than a 2048-env walk).
# Demand it FREE so an eval can still run on top without OOMing a run that has
# hours in it.
MIN_FREE_MIB=${MIN_FREE_MIB:-4500}

mkdir -p "$OUT" "$EVALDIR" logs
log() { echo "[$(date '+%F %T')] $*"; }

fingerprint() {
  .venv/bin/python - <<'PY'
from mjlab_microduck.train_hook import actuator_params_fingerprint
print(actuator_params_fingerprint()[0])
PY
}

FP_BEFORE=$(fingerprint)
log "=============================================================="
log "queue arm S2: $NAME   fingerprint $FP_BEFORE"
log "target: unfold the neck (neck_rel -1.934 in every phase, 32/32 draws)."
log "        This arm does NOT test the hesitating sit -- that mechanism is OPEN"
log "        (see the header: the 'collapsed composite head factor' explanation"
log "        was retracted; three non-multiplicative height terms price descent)."
log "iters $ITERS @ $ENVS envs   env: $ARM_ENV"
log "=============================================================="

[ -f "$RESULTS" ] || printf 'name\ttask\titers\tenvs\trun_dir\trc\tfp_before\tfp_after\telapsed_s\tonnx\teval_walk02\teval_zerocmd\teval_sitstand\n' > "$RESULTS"

# ── wait for a slot ─────────────────────────────────────────────────────────
waited=0
while :; do
  n=$(pgrep -f "train_cli Mjlab" | wc -l)
  free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
  # < 3 training procs AND 4500 MiB free: three 2048-env velocity runs
  # (3 x 2880 + ~1000 CUDA = 9.6 GB) leave only ~2.4 GB, which cannot host this
  # 4.4 GB run. So this arm correctly waits out one of them rather than OOMing it.
  if [ "$n" -lt 3 ] && [ "${free_mib:-0}" -ge "$MIN_FREE_MIB" ]; then
    log "[$NAME] slot free: $n procs, ${free_mib} MiB free (waited ${waited}s)"
    break
  fi
  [ $((waited % 900)) -eq 0 ] && log "[$NAME] waiting: $n procs, ${free_mib} MiB free (${waited}s)"
  if [ "$waited" -ge "$MAX_WAIT_S" ]; then
    log "[$NAME] gave up after ${MAX_WAIT_S}s without a slot"; exit 1
  fi
  sleep 60; waited=$((waited + 60))
done

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

# ── smoke ───────────────────────────────────────────────────────────────────
smoke_log="logs/smoke_${NAME}.log"
log "-- [$NAME] smoke (64 envs x ${SMOKE_ITERS}) with [$ARM_ENV]"
MICRODUCK_RUN_DIR_NAME="${NAME}_smoke" timeout -k 60 1800 env $ARM_ENV \
    .venv/bin/python -m mjlab_microduck.train_cli Mjlab-SitStand-Flat-MicroDuck \
    --env.scene.num-envs 64 --agent.max_iterations "$SMOKE_ITERS" > "$smoke_log" 2>&1
rc=$?
if [ "$rc" -ne 0 ] || grep -qE "RuntimeError|Traceback \(most recent" "$smoke_log"; then
  log "[$NAME] SMOKE FAILED rc=$rc -- not starting the run"
  grep -E "RuntimeError|Error|Traceback" "$smoke_log" | tail -3 | sed 's/^/   /'
  exit 1
fi
# The switch must actually have taken effect, or the arm is a silent duplicate of
# S1 and 6 GPU-hours buy nothing.
if grep -q "Episode_Reward/head_pose_l1" "$smoke_log"; then
  log "   ok: head_pose_l1 term IS active (the switch took effect)"
else
  log "   !! head_pose_l1 NOT in the smoke's reward log -- the switch did NOT take effect; aborting"
  grep -o "Episode_Reward/[a-z_0-9]*" "$smoke_log" | sort -u | sed 's/^/      /'
  exit 1
fi
# Every penalty must be <= 0 (the repo's infallible sign check).
bad=$(grep -oE "Episode_Reward/[a-z_0-9]+: -?[0-9.eE+-]+" "$smoke_log" | \
      awk '{v=$2+0; n=$1; sub("Episode_Reward/","",n); if (v>0 && n ~ /(penalt|l1$|limit|action_rate|torque_rate|ang_vel|momentum)/) print n"="$2}')
[ -n "$bad" ] && { log "   !! POSITIVE penalty value (double-negation bug): $bad"; exit 1; }

iter_t=$(grep -o "Iteration time: [0-9.]*s" "$smoke_log" | tail -1 | grep -o "[0-9.]*" || true)
[ -z "${iter_t:-}" ] && iter_t=3.0
est_s=$(python3 -c "print(int(float('${iter_t}')*${ITERS}*1.15))")
budget_s=$((est_s + BUDGET_SLACK_S))
log "   smoke ok: ${iter_t}s/iter (measured UNDER LOAD) -> est $(python3 -c "print(f'{${est_s}/3600:.1f}')")h, budget $(python3 -c "print(f'{${budget_s}/3600:.1f}')")h"

# ── train ───────────────────────────────────────────────────────────────────
started=$(date +%s)
MICRODUCK_RUN_DIR_NAME="$NAME" run_budgeted "$budget_s" "train_${NAME}.log" \
    env $ARM_ENV .venv/bin/python -m mjlab_microduck.train_cli \
    Mjlab-SitStand-Flat-MicroDuck \
    --env.scene.num-envs "$ENVS" --agent.max_iterations "$ITERS" \
    --agent.save_interval 1000
trc=$?
elapsed=$(( $(date +%s) - started ))

run_dir=$(grep -o "Logging experiment in directory: .*" "train_${NAME}.log" | tail -1 | sed 's/.*directory: //')
[ -z "${run_dir:-}" ] && run_dir="-"
onnx=$(ls -t "$run_dir"/*.onnx 2>/dev/null | head -1); [ -z "${onnx:-}" ] && onnx="-"
fp_after=$(fingerprint)
log "[$NAME] done rc=${trc} in $((elapsed/3600))h$(( (elapsed%3600)/60 ))m  run_dir=${run_dir}"
[ "$fp_after" != "$FP_BEFORE" ] && log "[$NAME] CONTAMINATED: $FP_BEFORE -> $fp_after"

# ── judge on a DISTRIBUTION, not one draw (S1's lesson) ─────────────────────
ev_sit="-"
if [ "$onnx" != "-" ]; then
  # The run dir's ONNX is re-exported at every save_interval, but verify it is at
  # least as new as the final checkpoint before trusting it.
  ck="$run_dir/model_${ITERS}.pt"
  [ -f "$ck" ] || ck=$(ls -t "$run_dir"/model_*.pt 2>/dev/null | head -1)
  if [ -f "$ck" ] && [ "$(stat -c %Y "$onnx")" -lt "$(stat -c %Y "$ck")" ]; then
    log "[$NAME] ONNX stale vs $(basename "$ck") -> re-exporting (bakes the normalizer)"
    .venv/bin/python scripts/export.py Mjlab-SitStand-Flat-MicroDuck \
        --checkpoint-file "$ck" --onnx-file "/tmp/${NAME}_final.onnx" \
        > logs/export_${NAME}.log 2>&1
    [ -f "/tmp/${NAME}_final.onnx" ] && onnx="/tmp/${NAME}_final.onnx"
  fi
  log "[$NAME] Q2 judged on ${SEEDS} seeded draws (a single draw is not a verdict)"
  OUT="$EVALDIR" ARMS="${onnx}:${NAME}" SEEDS="$SEEDS" MIN_FREE_MIB=1200 \
    bash scripts/sit_seed_sweep.sh >> "logs/${NAME}_seed_sweep.out" 2>&1
  [ -f "$EVALDIR/summary.tsv" ] && ev_sit="$EVALDIR/summary.tsv"
fi

printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
  "$NAME" "Mjlab-SitStand-Flat-MicroDuck" "$ITERS" "$ENVS" "$run_dir" "$trc" \
  "$FP_BEFORE" "$fp_after" "$elapsed" "$onnx" "-" "-" "$ev_sit" >> "$RESULTS"

{
  echo "== ${NAME} =="
  echo "env: $ARM_ENV   iters: ${ITERS} @ ${ENVS}"
  echo "status: rc=${trc}   elapsed: ${elapsed}s"
  echo "run_dir: ${run_dir}"
  echo "fingerprint: ${FP_BEFORE} -> ${fp_after}"
  echo "onnx: ${onnx}"
  echo "seed sweep: ${ev_sit}"
} >> "$STATUS"

log "=== $NAME FINISHED. Read logs/${NAME}_seed_sweep.out for the Q2 distribution. ==="
