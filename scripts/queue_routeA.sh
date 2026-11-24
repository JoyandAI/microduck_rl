#!/usr/bin/env bash
# ROUTE A queue (user decision 2026-09-18): align with the OFFICIAL GAIT
# (small steps, essentially no flight, small per-step yaw disturbance), accepting
# a slower robot. Budget: 3 seeds for the direction-finding test (~6-8 GPU-h).
#
# Arm: ours_flight = our recipe + head block + MICRODUCK_FLIGHT_PENALTY=1
#   ONE variable away from `ours_head`, which is already running (seed 42 default),
#   so seed 42 gives a PAIRED comparison against it and seeds 43/44 give the arm's
#   own spread. Seeds: 42, 43, 44.
#
# WHY THIS AND NOT "just lower the command": measured on a frozen checkpoint,
# sweeping only the commanded speed (scripts/probe_speed_vs_flight.sh) gives a
# PHASE TRANSITION at 0.10 -> 0.20 m/s -- per-step yaw impulse x93, foot clearance
# x12, net yaw 2 deg -> 114 deg. But below 0.10 the same policy does not move at
# all (made-good ~0.000 m/s, clearance 1-2 mm): it has NO walking mode, only
# "mark time" or "run". The official has the missing mode (clearance 12.5 mm, duty
# 0.585, flight 0.000, made-good 0.038 m/s). So the lever is a flight penalty, not
# a smaller number.
#
# Then, after the three seeds, the already-planned route-B probe (`ours_yawdc`,
# MICRODUCK_YAW_DC=1) runs so its pre-registered falsification test still lands.
#
# Usage: nohup setsid bash scripts/queue_routeA.sh > logs/queue_routeA.out 2>&1 &
set -u
cd "$(dirname "$0")/.." || exit 1
OUT=logs/queue_routeA
mkdir -p "$OUT"
export OUT
PRIOR=logs/queue_ours/status.txt

log() { printf '[%(%F %T)T] %s\n' -1 "$*" | tee -a "$OUT/status.txt" >&2; }
gpu_free_mib() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -dc '0-9'; }

printf 'label\ttask\titers\tenvs\trc\trun_dir\tfp_before\tfp_after\n' > "$OUT/results.tsv"

log '========================================================================'
log 'route A queue: ours_flight x 3 seeds (42/43/44), then ours_yawdc x 1'
log '========================================================================'

waited=0
while :; do
  marked=0; busy=0
  grep -q "queue complete" "$PRIOR" 2>/dev/null && marked=1
  pgrep -f "mjlab_microduck[.]train_cli" >/dev/null && busy=1
  if [ "$marked" -eq 1 ] && [ "$busy" -eq 0 ]; then
    log "prior queue done and GPU idle after ${waited}s"; break
  fi
  [ $((waited % 600)) -eq 0 ] && log "waiting: prior_marker=${marked} training_busy=${busy} (${waited}s)"
  sleep 60; waited=$((waited + 60))
done
free=$(gpu_free_mib || true)
while [ -z "$free" ] || [ "$free" -lt 9000 ]; do sleep 30; free=$(gpu_free_mib || true); done
log "GPU ready: ${free} MiB free"

OFF='MICRODUCK_STANCE_HIP_ROLL=0 MICRODUCK_ABDUCTION=0 MICRODUCK_YAW_SYMMETRY=0 MICRODUCK_SYMMETRY=0'
OURS='MICRODUCK_STANCE_HYGIENE=1 MICRODUCK_PER_JOINT_POSTURE=1 MICRODUCK_AIR_TIME_GATE=1'
OURS="$OURS MICRODUCK_HEADING_HOLD=1 MICRODUCK_HIP_ROLL_STD=0.12 MICRODUCK_POSE_ZERO_CMD_PROB=0.1"
OURS="$OURS MICRODUCK_HEAD_STABILITY=1 MICRODUCK_TURN_FIRST=0 MICRODUCK_YAW_DC=0"
FLIGHT="$OFF $OURS MICRODUCK_FLIGHT_PENALTY=1"
YAWDC="$OFF $OURS MICRODUCK_FLIGHT_PENALTY=0 MICRODUCK_YAW_DC=1"
WANT_FLIGHT="Episode_Reward/flight Episode_Reward/head_motion Metrics/yaw_dc_abs"
WANT_YAWDC="Episode_Reward/yaw_dc Metrics/yaw_dc_abs"

for s in 42 43 44; do
  bash scripts/run_arm_once.sh ours_flight "$s" "$FLIGHT" "$WANT_FLIGHT" || log "ours_flight s$s FAILED"
  pgrep -f "mjlab_microduck[.]train_cli" >/dev/null && { log "waiting for stragglers"; sleep 30; }
done

bash scripts/run_arm_once.sh ours_yawdc 42 "$YAWDC" "$WANT_YAWDC" || log "ours_yawdc FAILED"

log "route A queue complete; results: $OUT/results.tsv"
log "tables: $OUT/*_vx0.2.json (P64), $OUT/*_gait.json (duty/flight), $OUT/*_acceptance.log"
