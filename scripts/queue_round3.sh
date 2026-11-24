#!/usr/bin/env bash
# =============================================================================
# Queue round 3 (2026-09-16 night) — the two defects the USER reported, after the
# multi-agent attribution narrowed both to a specific, file-cited reward cause.
#
# ── Defect A: walking legs splayed to both sides ("有点像两侧迈开") ─────────────
# Measured (scripts/compare_walk_angles.py, 1 env, seed 0, frozen vx=0.2, 12 s):
#     official alpha_walking   L-R hip_roll, dev from HOME = 0.054 rad (~3 deg)
#     ours walk_hd1910_12000   L-R = 0.477 rad (~27 deg)
#     ours walk_hd1910_7000    L-R = 0.456 rad (~26 deg)
#   L+R ~ 0, so it is a mirror-symmetric SPLAY, not a lean.
# Cause (audited, file:line): the ONLY term that prices hip_roll is the pose
# Gaussian, and its std was relaxed 0.05 -> 0.12 on 2026-09-16. The OFFICIAL
# recipe uses 0.05, and its comment says exactly what it is for:
#     "hold the 5-degree-inward stance (sole sits flat), stop leg splay"
#     (tasks/reference_official_velocity_env_cfg.py:225,234)
# Everything else is blind to a STATIC offset: action_rate/body_ang_vel are rate
# terms (a constant offset costs them exactly zero), dof_pos_limits' soft limit
# sits at +-0.346 rad while the observed |q| is 0.154, action_over_limit needs
# |action| > 0.77 while the observed is 0.24, and stance_hip_yaw's joint selector
# matches hip_yaw only -- it cannot see hip_roll.
#   A1 splay_std05 : MICRODUCK_HIP_ROLL_STD=0.05   the official value, one change
#   A2 splay_hrl1  : MICRODUCK_STANCE_HIP_ROLL=1   an L1 on hip_roll (the shape
#                    the rollers env already used to cure the same defect)
#   A3 splay_both  : both
#
# ── Defect B: sit-stand head still droops (user: "站起来的头部还是向下垂的") ──
# Measured (scripts/compare_sitstand_angles.py, same env/seed/schedule):
#     official alpha_sitstand  neck_pitch dev -0.011 rad, head height 229.4 mm
#     ours (delivered)         neck_pitch dev -1.932,     head height  87.7 mm
#     ours (head-ground fix)   neck_pitch dev -1.934,     head height 153.2 mm
#   i.e. the previous fix moved the head 88 -> 153 mm by straightening the TRUNK
#   and did not move the NECK AT ALL. Nominal head height is 232.6 mm.
# Cause: the two head-facing terms that could specify a target both have a
# vanishing gradient at a 1.93 rad error -- head_pose_tracking is a per-joint
# Gaussian averaged over 4 joints (gradient ~1e-6 /rad there) and the head factor
# of posture_composite is multiplicative and has collapsed to ~0.003 (0.028 /rad).
# The limit terms (neck_limit_proximity + dof_pos_limits) give a bounded
# 0.35/s lump over the first 0.21 rad and then EXACTLY ZERO for the remaining
# ~1.5 rad. The one term shaped to survive a large error -- an L1 that is SUMmed
# (not averaged) and does not vanish -- is `head_pose_l1`, implemented, unit
# tested and calibrated at 22.7 % of the reward stack, and behind a switch that
# was never trained.
#   B1 sit_head_l1 : MICRODUCK_HEAD_L1=1
#
# Safety rails, inherited: actuator fingerprint before/after, 64-env smoke test
# first, budget from the smoke's OWN iteration time measured UNDER LOAD, VRAM
# gate on FREE memory (not process count).
#
# Usage:
#   setsid nohup bash scripts/queue_round3.sh > logs/queue_r3.out 2>&1 < /dev/null &
# =============================================================================
set -u
cd "$(dirname "$0")/.." || exit 1
export UV_CACHE_DIR=/tmp/uv-cache
export WANDB_MODE=offline

OUT=logs/queue_round3
STATUS=logs/queue_r3_status.txt
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
log "=================================================================="
log "queue round 3: A) walking splay  B) sit-stand head droop"
log "fingerprint $FP_BEFORE"
log "arms: splay_std05 -> sit_head_l1 -> splay_hrl1 -> splay_both"
log "=================================================================="

[ -f "$RESULTS" ] || printf 'name\ttask\titers\tenvs\trun_dir\trc\tfp_before\tfp_after\telapsed_s\tonnx\tsplay\thead\n' > "$RESULTS"

wait_for_slot() {  # $1 = label, $2 = MiB needed
  local label="$1" need="$2" waited=0 n free_mib
  while :; do
    n=$(pgrep -f "train_cli Mjlab" | wc -l)
    free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    if [ "${free_mib:-0}" -ge "$need" ]; then
      log "[$label] slot free: ${free_mib} MiB free (need ${need}), ${n} running (waited ${waited}s)"
      return 0
    fi
    [ $((waited % 900)) -eq 0 ] && log "[$label] waiting: ${free_mib} MiB free (need ${need}), ${n} running (${waited}s)"
    [ "$waited" -ge "$MAX_WAIT_S" ] && { log "[$label] gave up after ${MAX_WAIT_S}s"; return 1; }
    sleep 60; waited=$((waited + 60))
  done
}

run_budgeted() {  # $1 budget, $2 out log, rest = command
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

# $1 name, $2 task, $3 iters, $4 envs, $5 MiB needed, $6 env assignments
run_arm() {
  local name="$1" task="$2" iters="$3" envs="$4" need="$5" arm_env="$6"
  local smoke_log="logs/r3_smoke_${name}.log" train_log="logs/r3_train_${name}.log"
  wait_for_slot "$name" "$need" || return 1

  log "-- [$name] smoke (64 envs x ${SMOKE_ITERS}) [$arm_env]"
  MICRODUCK_RUN_DIR_NAME="${name}_smoke" timeout -k 60 1800 env $arm_env \
      .venv/bin/python -m mjlab_microduck.train_cli "$task" \
      --env.scene.num-envs 64 --agent.max_iterations "$SMOKE_ITERS" > "$smoke_log" 2>&1
  local rc=$?
  if [ "$rc" -ne 0 ] || grep -qE "RuntimeError|Traceback \(most recent" "$smoke_log"; then
    log "[$name] SMOKE FAILED rc=$rc -- skipping"
    grep -E "RuntimeError|Error|Traceback" "$smoke_log" | tail -3 | sed 's/^/   /'
    return 1
  fi
  # The switch must actually have taken effect, or the arm is a silent duplicate
  # of the baseline and the GPU-hours buy nothing.
  case "$name" in
    sit_head_l1) grep -q "Episode_Reward/head_pose_l1" "$smoke_log" \
        && log "   ok: head_pose_l1 term IS active" \
        || { log "   !! head_pose_l1 ABSENT -- switch did not take effect"; return 1; } ;;
    splay_hrl1|splay_both) grep -q "Episode_Reward/stance_hip_roll" "$smoke_log" \
        && log "   ok: stance_hip_roll term IS active" \
        || { log "   !! stance_hip_roll ABSENT -- switch did not take effect"; return 1; } ;;
  esac
  # Every penalty-signed term must be <= 0 (the repo's infallible sign check).
  local bad
  bad=$(grep -oE "Episode_Reward/[a-z_0-9]+: -?[0-9.eE+-]+" "$smoke_log" | \
        awk '{v=$2+0; n=$1; sub("Episode_Reward/","",n);
              if (v>0 && n ~ /(penalt|l1$|limit|action_rate|torque_rate|ang_vel|momentum|slip|clearance|no_stepping|neck_motion)/) print n"="$2}')
  [ -n "$bad" ] && { log "   !! POSITIVE penalty (double-negation bug): $bad"; return 1; }

  local iter_t est_s budget_s
  iter_t=$(grep -o "Iteration time: [0-9.]*s" "$smoke_log" | tail -1 | grep -o "[0-9.]*" || true)
  [ -z "${iter_t:-}" ] && iter_t=3.0
  est_s=$(python3 -c "print(int(float('${iter_t}')*${iters}*1.15))")
  budget_s=$((est_s + BUDGET_SLACK_S))
  log "   smoke ok: ${iter_t}s/iter (UNDER LOAD) -> est $(python3 -c "print(f'{${est_s}/3600:.1f}')")h, budget $(python3 -c "print(f'{${budget_s}/3600:.1f}')")h"

  local started; started=$(date +%s)
  MICRODUCK_RUN_DIR_NAME="$name" run_budgeted "$budget_s" "$train_log" \
      env $arm_env .venv/bin/python -m mjlab_microduck.train_cli "$task" \
      --env.scene.num-envs "$envs" --agent.max_iterations "$iters" \
      --agent.save_interval 1000
  local trc=$? elapsed=$(( $(date +%s) - started ))

  local run_dir onnx fp_after
  run_dir=$(grep -o "Logging experiment in directory: .*" "$train_log" | tail -1 | sed 's/.*directory: //')
  [ -z "${run_dir:-}" ] && run_dir="-"
  onnx=$(ls -t "$run_dir"/*.onnx 2>/dev/null | head -1); [ -z "${onnx:-}" ] && onnx="-"
  fp_after=$(fingerprint)
  log "[$name] done rc=${trc} in $((elapsed/3600))h$(( (elapsed%3600)/60 ))m run_dir=${run_dir}"
  [ "$fp_after" != "$FP_BEFORE" ] && log "[$name] CONTAMINATED: $FP_BEFORE -> $fp_after"

  # ── the measurement that answers the user's complaint, not a proxy ─────────
  local ev_splay="-" ev_head="-"
  if [ "$onnx" != "-" ]; then
    local ck="$run_dir/model_${iters}.pt"
    [ -f "$ck" ] || ck=$(ls -t "$run_dir"/model_*.pt 2>/dev/null | head -1)
    if [ -f "$ck" ] && [ "$(stat -c %Y "$onnx")" -lt "$(stat -c %Y "$ck")" ]; then
      log "[$name] ONNX stale vs $(basename "$ck") -> re-exporting"
      .venv/bin/python scripts/export.py "$task" --checkpoint-file "$ck" \
          --onnx-file "/tmp/r3_${name}.onnx" > "logs/r3_export_${name}.log" 2>&1
      [ -f "/tmp/r3_${name}.onnx" ] && onnx="/tmp/r3_${name}.onnx"
    fi
    if [ "${task}" = "Mjlab-Velocity-Flat-MicroDuck" ]; then
      .venv/bin/python scripts/compare_walk_angles.py \
          --policy "official:/home/joyandai/microduck/policies/alpha_walking.onnx:preswap" \
          --policy "$name:$onnx" --vx 0.2 --seed 0 \
          > "$OUT/${name}_splay.log" 2>&1 && ev_splay="$OUT/${name}_splay.log"
      .venv/bin/python scripts/eval_straightness.py --onnx "$onnx" --task "$task" \
          --lin-vel-x 0.2 --num-envs 64 --seed 0 --seconds 20 --freeze-pose-cmd \
          --json "$OUT/${name}_walk02.json" > "$OUT/${name}_walk02.log" 2>&1
    else
      .venv/bin/python scripts/compare_sitstand_angles.py \
          --policy "official:/home/joyandai/microduck/policies/alpha_sitstand.onnx:preswap" \
          --policy "$name:$onnx" --seed 0 > "$OUT/${name}_head.log" 2>&1 \
        && ev_head="$OUT/${name}_head.log"
    fi
  fi

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$name" "$task" "$iters" "$envs" "$run_dir" "$trc" \
    "$FP_BEFORE" "$fp_after" "$elapsed" "$onnx" "$ev_splay" "$ev_head" >> "$RESULTS"

  {
    echo "== ${name} =="
    echo "env: $arm_env   iters: ${iters} @ ${envs}"
    echo "status: rc=${trc}   elapsed: ${elapsed}s"
    echo "run_dir: ${run_dir}   onnx: ${onnx}"
    echo "fingerprint: ${FP_BEFORE} -> ${fp_after}"
    echo "splay log: ${ev_splay}   head log: ${ev_head}"
    # print the headline numbers inline so the morning read is one file
    [ "$ev_splay" != "-" ] && grep -A4 "hip roll/yaw" "$ev_splay" | sed 's/^/   /'
    [ "$ev_head" != "-" ] && grep -A4 "head joints the eye" "$ev_head" | sed 's/^/   /'
  } >> "$STATUS"
}

# ORDER: the two highest-confidence fixes first (they answer the user's two
# complaints), then the two isolation arms. Each waits for its own VRAM, so this
# self-schedules around whatever else is still running.
# The VRAM thresholds are set to CAP CONCURRENCY AT TWO of these arms, not at
# "whatever fits". Measured per-run cost: velocity@2048 2880 MiB, sitstand@1024
# 4406 MiB, plus ~1000 MiB of CUDA/display. Two arms = 7.3 GB used / ~4 GB free,
# which is the mix this repo has actually measured as safe. Admitting a THIRD
# (2 velocity + 1 sitstand = 11.2 GB) would leave ~1 GB, and an OOM there kills a
# run that already has hours in it -- the most expensive failure this queue has.
# So a velocity arm demands 4300 MiB free (2880 + 1.4 GB headroom) and the
# sitstand arm 5000; the third arm then WAITS for a real slot instead of
# squeezing in.
run_arm splay_std05 Mjlab-Velocity-Flat-MicroDuck   9000 2048 4300 "MICRODUCK_HIP_ROLL_STD=0.05"
run_arm sit_head_l1 Mjlab-SitStand-Flat-MicroDuck   8000 1024 5000 "MICRODUCK_HEAD_L1=1"
run_arm splay_both  Mjlab-Velocity-Flat-MicroDuck   9000 2048 4300 "MICRODUCK_HIP_ROLL_STD=0.05 MICRODUCK_STANCE_HIP_ROLL=1"
run_arm splay_hrl1  Mjlab-Velocity-Flat-MicroDuck   9000 2048 4300 "MICRODUCK_STANCE_HIP_ROLL=1"

log "=================================================================="
log "queue round 3 finished. status: $STATUS"
log "splay: logs/queue_round3/*_splay.log   head: logs/queue_round3/*_head.log"
