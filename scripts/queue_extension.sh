#!/usr/bin/env bash
# EXTENSION queue — runs after scripts/queue_overnight.sh finishes.
#
# WHAT THE OVERNIGHT RUN ESTABLISHED (2026-09-20):
#  * The flight penalty WORKS on its own metric: flight 0.229 -> 0.017/0.033/0.043
#    (mean 0.031 +- 0.013), 3/3 seeds, 7.4x, with speed 0.004 -> 0.045 and path
#    efficiency 0.087 -> 0.165-0.197. First mechanism-targeted lever with a
#    consistent cross-seed effect. Measurement self-checked against agent D's
#    published numbers (official 0.000 / ck7000 0.120).
#  * ... but it did NOT buy straightness. Across EVERY arm measured, flight does not
#    rank the wander (ours_flight_s43 has the 2nd-lowest flight AND the 2nd-worst net
#    yaw; turn_official has 0.24 flight and the best net yaw of the new arms).
#  * The per-half-step yaw impulse (agent D's causal quantity; harness self-checked:
#    ck7000 6.35 = D's 6.35) does not rank it either: ours_head_s43 generates only
#    0.64 deg/half-step (official 0.36, align_fwd 0.63) yet accumulates +592 deg.
#  * What DOES separate us is the RANDOM WALK: per-env net yaw over 20 s has median
#    78-118 deg for our arms vs 2.8 deg for the official, while the SIGNED mean
#    cancels (ck7000 mean -14 deg, median 118 deg) -- spontaneous symmetry breaking.
#    The official's per-step net is ~0.09 deg; ours ranges 0.1-6 deg WITH RANDOM SIGN
#    across envs.
#
# => The quantity to attack is each env's DRIFT (the random walk), and the only
#    implemented term that prices exactly that is `yaw_dc` (L1 on EMA_tau(wz) per env,
#    gated to straight commands): |EMA| IS the local drift rate, which is what the
#    accumulated 20 s wander integrates. That is why this extension spends its seeds
#    on `ours_yawdc`, head-to-head with the flight arm's 3 seeds, instead of buying
#    8 seeds of the flight arm (whose mechanism is measured not to matter here).
#
# Arms added:
#   ours_yawdc      s43, s44   -> 3-seed mean comparable with ours_flight's 3 seeds
#   ours_flight_yawdc s42      -> do the two mechanisms compose?
#
# Pre-registered criterion (agent A): if |EMA| (logged as Metrics/yaw_dc_abs) stays
# >0.10 rad/s ~1000 iterations after the 300-iteration ramp, the "under-priced"
# hypothesis is FALSIFIED and the drift is not a pricing problem. run_arm_once prints
# that number from the run's own log.
#
# Usage: nohup setsid bash scripts/queue_extension.sh > logs/queue_extension.out 2>&1 &
set -u
cd "$(dirname "$0")/.." || exit 1
OUT=logs/queue_extension
mkdir -p "$OUT"
export OUT
PRIOR=logs/queue_overnight/status.txt

log() { printf '[%(%F %T)T] %s\n' -1 "$*" | tee -a "$OUT/status.txt" >&2; }
gpu_free_mib() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -dc '0-9'; }

printf 'label\ttask\titers\tenvs\trc\trun_dir\tfp_before\tfp_after\n' > "$OUT/results.tsv"

waited=0
while :; do
  marked=0; busy=0
  grep -q "OVERNIGHT QUEUE COMPLETE" "$PRIOR" 2>/dev/null && marked=1
  pgrep -f "mjlab_microduck[.]train_cli" >/dev/null && busy=1
  if [ "$marked" -eq 1 ] && [ "$busy" -eq 0 ]; then
    log "prior queue complete and GPU idle after ${waited}s"; break
  fi
  [ $((waited % 600)) -eq 0 ] && log "waiting: prior_marker=${marked} busy=${busy} (${waited}s)"
  sleep 60; waited=$((waited + 60))
done
free=$(gpu_free_mib || true)
while [ -z "$free" ] || [ "$free" -lt 9000 ]; do sleep 30; free=$(gpu_free_mib || true); done
log "GPU ready: ${free} MiB free"

OFF='MICRODUCK_STANCE_HIP_ROLL=0 MICRODUCK_ABDUCTION=0 MICRODUCK_YAW_SYMMETRY=0 MICRODUCK_SYMMETRY=0'
OURS='MICRODUCK_STANCE_HYGIENE=1 MICRODUCK_PER_JOINT_POSTURE=1 MICRODUCK_AIR_TIME_GATE=1'
OURS="$OURS MICRODUCK_HEADING_HOLD=1 MICRODUCK_HIP_ROLL_STD=0.12 MICRODUCK_POSE_ZERO_CMD_PROB=0.1"
HEAD="$OFF $OURS MICRODUCK_HEAD_STABILITY=1 MICRODUCK_TURN_FIRST=0"

for s in 43 44; do
  bash scripts/run_arm_once.sh ours_yawdc "$s" \
    "$HEAD MICRODUCK_YAW_DC=1" \
    "Episode_Reward/yaw_dc Metrics/yaw_dc_abs" || log "ours_yawdc s$s FAILED"
done

bash scripts/run_arm_once.sh ours_flight_yawdc 42 \
  "$HEAD MICRODUCK_FLIGHT_PENALTY=1 MICRODUCK_YAW_DC=1" \
  "Episode_Reward/flight Episode_Reward/yaw_dc Metrics/yaw_dc_abs" \
  || log "ours_flight_yawdc FAILED"

log "EXTENSION QUEUE COMPLETE; results: $OUT/results.tsv"
