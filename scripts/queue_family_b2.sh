#!/usr/bin/env bash
# FAMILY B, ROUND 2 — the two defects the round-1 report exposed, plus the checkpoint
# sweep that family B never got.
#
# What round 1 (famB_head_flight, official recipe + head + flight, 12000 it) achieved:
#   60/64 envs walking (best ever; align_fwd 51), walkers 50.1 deg / 21.5 deg-per-metre
#   (align_fwd 64.3 / 27.3, official 11.1), path efficiency 0.663 (best ever), 0.1166 m/s,
#   0 falls, standstill 1 reset / 5.6 deg tilt.
#
# DEFECT 1 — standstill marching: clearance 14.65 mm at zero command, above the repo's
# 8 mm bar (standing 0.5-4.2 mm, marching 19.4 mm). Root cause candidate, and it is a
# one-variable change: the OFFICIAL recipe has NO zero-command bucket
# (`MICRODUCK_POSE_ZERO_CMD_PROB=0.0`, `rel_standing_envs` only), and AGENTS.md already
# records that "uniform sampling essentially never produces the all-zero command, which
# is exactly the deployment idle state". The deviant recipe trained exactly that bucket
# (0.1) and its flight arm stood at 0.51 mm — the cleanest standstill we have ever
# measured. So: add the bucket.
#
# DEFECT 2 — the other standstill lever, measured: `ours_flight_stance` (deviant recipe
# + flight + MICRODUCK_FEET_MAX_WIDTH=1) stood at 1 reset / 12.1 deg / 1.07 mm, i.e.
# feet_max_width is the one switch that ever produced a clean zero-command stance when
# combined with flight. It wrecked nothing else in that arm except straightness, which
# family B supplies. So test it on family B too.
#
# Plus: family B never went through queue_ckpt_sweep_arms.sh (it was trained after that
# sweep ran), and the sweep is what found that align_full's best checkpoint is 73.7 deg
# where its final is 199 deg — i.e. straightness is not monotone in training time and
# the best deliverable is usually a MID-RUN checkpoint.
#
# Usage: nohup setsid bash scripts/queue_family_b2.sh > logs/queue_family_b2.out 2>&1 &
set -u
cd "$(dirname "$0")/.." || exit 1
OUT=logs/queue_family_b2
mkdir -p "$OUT"
export OUT
ITERS=${ITERS:-12000}
export ITERS

log() { printf '[%(%F %T)T] %s\n' -1 "$*" | tee -a "$OUT/status.txt" >&2; }
gpu_free_mib() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -dc '0-9'; }

# ── gate: family B round 1 owns the GPU ─────────────────────────────────────
waited=0
while :; do
  b1=$(grep -c "FAMILY B QUEUE COMPLETE" logs/queue_family_b/status.txt 2>/dev/null || true)
  busy=$(pgrep -f "mjlab_microduck[.]train_cli" | wc -l)
  [ "${b1:-0}" -gt 0 ] && [ "$busy" -eq 0 ] && { log "family B round 1 done; taking the GPU"; break; }
  [ $((waited % 600)) -eq 0 ] && log "waiting: b1_marker=${b1:-0} training=${busy} (${waited}s)"
  sleep 60; waited=$((waited + 60))
done

ALIGN_OFF='MICRODUCK_STANCE_HYGIENE=0 MICRODUCK_STANCE_HIP_ROLL=0 MICRODUCK_PER_JOINT_POSTURE=0'
ALIGN_OFF="$ALIGN_OFF MICRODUCK_AIR_TIME_GATE=0 MICRODUCK_HEADING_HOLD=0 MICRODUCK_YAW_HOLD_ZERO_CMD=0"
ALIGN_OFF="$ALIGN_OFF MICRODUCK_ABDUCTION=0 MICRODUCK_YAW_DC=0 MICRODUCK_YAW_SYMMETRY=0"
ALIGN_OFF="$ALIGN_OFF MICRODUCK_TURN_FIRST=0 MICRODUCK_SYMMETRY=0 MICRODUCK_HIP_ROLL_STD=0.05"
ALIGN_OFF="$ALIGN_OFF MICRODUCK_POSE_ZERO_CMD_PROB=0.0"
HEAD_OFFICIAL="$ALIGN_OFF MICRODUCK_HEAD_STABILITY=1"
W="Episode_Reward/flight Metrics/air_time_mean Metrics/yaw_dc_abs"

# C1/C2: the zero-command bucket on the winning arm, two seeds (defect 1).
for s in 42 43; do
  bash scripts/run_arm_once.sh famB_flight_zerocmd "$s" \
    "$HEAD_OFFICIAL MICRODUCK_FLIGHT_PENALTY=1 MICRODUCK_POSE_ZERO_CMD_PROB=0.1" \
    "$W" || log "famB_flight_zerocmd s$s FAILED"
done

# C3: the feet_max_width lever (defect 2).
bash scripts/run_arm_once.sh famB_flight_feet 42 \
  "$HEAD_OFFICIAL MICRODUCK_FLIGHT_PENALTY=1 MICRODUCK_FEET_MAX_WIDTH=1" \
  "$W Episode_Reward/feet_max_width" || log "famB_flight_feet FAILED"

# ── checkpoint sweep over family B (it never got one) ───────────────────────
# Straightness is not monotone in training time: align_full scores 73.7 deg at
# ck03000 and 199 deg at the end. Best checkpoint per arm is the KPI.
log "starting the family-B checkpoint sweep"
SWEEP_OUT=logs/queue_sweep_famB SKIP_WAIT=1 \
# Run-dir names come from run_arm_once.sh as "<label>_s<seed>", so every seed suffix
# is _s42 / _s43 — the round-1 queue's own labels are listed the same way.
SWEEP_ARMS="famB_head_flight_s42 famB_head_flight_s43 famB_head_flight_s44 famB_head_s43 famB_head_s44 famB_flight_zerocmd_s42 famB_flight_zerocmd_s43 famB_flight_feet_s42 famB_head_flight_slow_s42 famB_head_flight_yawdc_s42" \
  bash scripts/queue_ckpt_sweep_arms.sh || log "family-B sweep failed"

log "FAMILY B ROUND 2 COMPLETE; results: $OUT/results.tsv ; sweep: logs/queue_sweep_famB/summary.txt"
