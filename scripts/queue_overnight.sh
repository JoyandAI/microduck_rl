#!/usr/bin/env bash
# OVERNIGHT + NEXT-DAY MASTER QUEUE (user: "我今天晚上和明天不会用这个电脑…尽量排满").
#
# Serial on purpose. MEASURED: one velocity@2048 run alone iterates at ~1.05 s/iter
# (align_fwd: 9000 iters in 2.6 h), while two concurrent runs each go to ~2.9 s/iter
# = 1.45 s/iter of effective per-run cost. So serial fits ~9 runs in 24 h against
# ~6.6 for two-wide -- serial is both faster in total and lower-variance.
#
# ORDER = information per GPU-hour, from the six agent reports:
#
#  1. turn_ours      x2  (F's single prescription)  our recipe, head terms OFF,
#                        MICRODUCK_TURN_FIRST=1.  F measured the steering gain at
#                        11.3 % of command (official 57.6 %) and showed that weight
#                        changes cannot fix a signal/noise ratio set by the gait:
#                        only the BASIN can change, and a turn-only curriculum is
#                        the one basin-changing lever that was never falsified.
#                        Head terms are OFF because the acceptance table shows they
#                        covary with hip_yaw being welded to its stop 99.5 % of the
#                        walk (vs 46 % without them, ~0 % for the official), and a
#                        welded joint has exactly zero steering gain.
#  2. ours_flight    x3  (route A, user's choice)   the flight penalty: official
#                        flight = 0.000 (64/64), ours 0.119-0.235, and the measured
#                        speed sweep shows a PHASE TRANSITION at 0.10->0.20 m/s
#                        (per-step yaw impulse x93). 3 seeds so the mean+SD is
#                        comparable with the 3-seed baseline below.
#  3. ours_head s43,s44  completing the 3-seed BASELINE (s42 is running now), which
#                        is what makes the flight arm's numbers interpretable.
#  4. ours_stance    x1  feet_max_width (D's prescription): ours 122-133 mm vs the
#                        official's 65-84 mm; the existing feet_distance penalty is
#                        the wrong direction (it charges feet too CLOSE) and is
#                        exactly 0.0 at our operating point.
#  5. ours_ztrack    x1  de-diluted yaw-rate tracking (A's 4.2x): the official term
#                        puts the unavoidable roll/pitch rates in the same exponent
#                        as the yaw error, costing the yaw channel ~4.4x authority.
#  6. ours_flight_stance x1  the two route-A levers together.
#  7. ours_yawdc     x1  route-B probe (L1 on the EMA of wz) with A's pre-registered
#                        falsification: |EMA| still >0.10 after the ramp => the
#                        "under-priced" hypothesis dies and reachability is all
#                        that is left.
#
# Every arm gets: smoke + switch verification, 9000 iters @2048, ONNX export, P64
# walk + stand evals, gait forensics (duty/flight), and the 17-row acceptance table.
# The live sweep (scripts/queue_align_sweep_live.sh) covers the checkpoints.
#
# Usage: nohup setsid bash scripts/queue_overnight.sh > logs/queue_overnight.out 2>&1 &
set -u
cd "$(dirname "$0")/.." || exit 1
OUT=logs/queue_overnight
mkdir -p "$OUT"
export OUT

log() { printf '[%(%F %T)T] %s\n' -1 "$*" | tee -a "$OUT/status.txt" >&2; }
gpu_free_mib() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -dc '0-9'; }

# ── serial gate: nothing else training, and the driver is alive ──────────────
waited=0
while :; do
  busy=$(pgrep -f "mjlab_microduck[.]train_cli" | wc -l)  # includes the adopted run
  free=$(gpu_free_mib || true)
  if [ "$busy" -eq 0 ] && [ -n "$free" ] && [ "$free" -ge 9000 ]; then
    log "GPU idle (${free} MiB free) after ${waited}s"; break
  fi
  [ $((waited % 600)) -eq 0 ] && log "waiting: training_procs=${busy} free=${free:-?} (${waited}s)"
  sleep 60; waited=$((waited + 60))
done

# 0. adopt the orphaned ours_head s42 (its driver was stopped on purpose): wait for
#    it to finish, then export + P64 eval + acceptance, so the 3-seed baseline is
#    complete (s43/s44 are re-run below).
bash scripts/adopt_orphan_run.sh ours_head logs/rsl_rl/velocity/ours_head \
  || log "ours_head s42 adoption failed"

OFF='MICRODUCK_STANCE_HIP_ROLL=0 MICRODUCK_ABDUCTION=0 MICRODUCK_YAW_SYMMETRY=0 MICRODUCK_SYMMETRY=0'
OURS='MICRODUCK_STANCE_HYGIENE=1 MICRODUCK_PER_JOINT_POSTURE=1 MICRODUCK_AIR_TIME_GATE=1'
OURS="$OURS MICRODUCK_HEADING_HOLD=1 MICRODUCK_HIP_ROLL_STD=0.12 MICRODUCK_POSE_ZERO_CMD_PROB=0.1"
BASE="$OFF $OURS"                       # our recipe (12 deviations), no head terms
HEAD="$BASE MICRODUCK_HEAD_STABILITY=1" # + the head block (the user's ask)

# 1a. turn-first on OUR recipe without the head terms (the base that produced our
#     best walker ck7000, so the forward gait can form once the curriculum widens)
for s in 42 43; do
  STEER_PROBE=1 bash scripts/run_arm_once.sh turn_ours "$s" \
    "$BASE MICRODUCK_HEAD_STABILITY=0 MICRODUCK_TURN_FIRST=1" \
    "Episode_Reward/stance_hip_yaw turn_first" || log "turn_ours s$s FAILED"
done
# 1b. turn-first on the OFFICIAL core, head terms off -- F's literal prescription
#     (stage 0 asks for turning only, lin +-0.05, so a walking base is not needed
#     to discover the skill; the point is to find it BEFORE a forward limit cycle)
for s in 42 43; do
  STEER_PROBE=1 bash scripts/run_arm_once.sh turn_official "$s" \
    "$OFF MICRODUCK_STANCE_HYGIENE=0 MICRODUCK_PER_JOINT_POSTURE=0 MICRODUCK_AIR_TIME_GATE=0 MICRODUCK_HEADING_HOLD=0 MICRODUCK_HIP_ROLL_STD=0.05 MICRODUCK_POSE_ZERO_CMD_PROB=0.0 MICRODUCK_HEAD_STABILITY=0 MICRODUCK_TURN_FIRST=1" \
    "turn_first" || log "turn_official s$s FAILED"
done

# 2. route A: flight penalty WITH the head terms (the user's default family), 3 seeds
for s in 42 43 44; do
  bash scripts/run_arm_once.sh ours_flight "$s" \
    "$HEAD MICRODUCK_TURN_FIRST=0 MICRODUCK_FLIGHT_PENALTY=1" \
    "Episode_Reward/flight Episode_Reward/head_motion Metrics/yaw_dc_abs" || log "ours_flight s$s FAILED"
done

# 3. baseline seeds (s42 is the run in flight now) -> 3-seed baseline
for s in 43 44; do
  bash scripts/run_arm_once.sh ours_head "$s" \
    "$HEAD MICRODUCK_TURN_FIRST=0" \
    "Episode_Reward/head_motion Metrics/yaw_dc_abs" || log "ours_head s$s FAILED"
done

# 4-7. single-seed probes of the remaining levers
bash scripts/run_arm_once.sh ours_stance 42 \
  "$HEAD MICRODUCK_TURN_FIRST=0 MICRODUCK_FEET_MAX_WIDTH=1" \
  "Episode_Reward/feet_max_width Metrics/yaw_dc_abs" || log "ours_stance FAILED"

bash scripts/run_arm_once.sh ours_ztrack 42 \
  "$HEAD MICRODUCK_TURN_FIRST=0 MICRODUCK_YAW_Z_TRACK=1" \
  "Episode_Reward/track_angular_velocity_z Metrics/yaw_dc_abs" || log "ours_ztrack FAILED"

bash scripts/run_arm_once.sh ours_flight_stance 42 \
  "$HEAD MICRODUCK_TURN_FIRST=0 MICRODUCK_FLIGHT_PENALTY=1 MICRODUCK_FEET_MAX_WIDTH=1" \
  "Episode_Reward/flight Episode_Reward/feet_max_width" || log "ours_flight_stance FAILED"

bash scripts/run_arm_once.sh ours_yawdc 42 \
  "$HEAD MICRODUCK_TURN_FIRST=0 MICRODUCK_YAW_DC=1" \
  "Episode_Reward/yaw_dc Metrics/yaw_dc_abs" || log "ours_yawdc FAILED"

log "OVERNIGHT QUEUE COMPLETE; results: $OUT/results.tsv"
log "per-arm tables under $OUT/ ; sweep: logs/queue_align/sweep/"
