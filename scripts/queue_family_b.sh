#!/usr/bin/env bash
# FAMILY B — the recipe the evidence actually favours, 3x budget, 3 seeds per arm.
#
# WHY NOT THE FAMILY THE PREVIOUS QUEUES RAN (2026-09-20 decision):
# Every arm in queue_overnight / queue_routeA / queue_extension is our
# 12-deviation recipe ("$HEAD"), and measured on ONE protocol (64 env, 20 s,
# frozen vx=0.2, the same physics) that family is the worst of everything we have:
#
#   policy                                   walkers   |net yaw|   speed   zero-cmd
#   official recipe VERBATIM (align_official)   7/64     405.6      0.00     -
#   official recipe + head (align_full)        24/64      64.2      0.00     0 falls
#   official recipe + head (align_fwd)         51/64      64.3      0.118    1 reset
#   ck7000 (our old recipe)                    49/64      90.7      0.089    1 reset
#   ours_head s42/s43/s44 (queue family)    18-30/64  383.6-452.1   ~0.05    62-64 resets
#   ours_stance s42 (just trained)              -        213.1      0.034    55 resets / 86 mm
#
# i.e. the official recipe + head block walks straighter (64 vs 384-452 deg), faster
# (0.118 vs 0.05 m/s) and stands still (1 reset vs 55-64) — it is better on every
# axis, and it is the configuration the user asked to align with in the first place.
# The 12 deviations were added to patch symptoms of an earlier policy; the data now
# says they are the problem, not the fix. Reproduce the table with
# scripts/report_queue_arms.py.
#
# WHAT THIS QUEUE TESTS, in order of information per GPU-hour:
#  B1  famB_head_flight   x3 seeds — the flight penalty was the ONE lever with a
#      triple win on the deviant recipe (flight 0.229 -> 0.017/0.033/0.043; the
#      zero-command collapse 62-64 resets -> 1; yaw DC 22.9 -> 5.6 deg/s). Nobody
#      has ever run it on the good family. This is the primary arm.
#  B2  famB_head          x2 seeds — the control (same config as align_full /
#      align_fwd, which exist but were run without a recorded seed, and which
#      differ from each other by 2x in walker fraction), so the flight A/B is
#      3 seeds vs 3 seeds on the same protocol.
#  B3  famB_head_flight_slow x1 — the SAME as B1 plus MICRODUCK_CMD_LIN_MAX=0.15:
#      the frozen-ck7000 speed sweep found a PHASE TRANSITION between 0.10 and
#      0.20 m/s (per-step yaw impulse x93, net yaw 2 -> 114 deg), so "is our gait
#      only straight when it is slow?" is a one-variable question. It also answers
#      the user's own 2026-09-18 hypothesis ("是不是速度太快导致的").
#  B4  famB_head_flight_yawdc x1 — conditional, included so the GPU is never idle:
#      if ours_yawdc (queue_overnight) clears its pre-registered bar, add the term;
#      if it fails, this arm is cancelled and its slot goes to a B1 seed-45 rerun.
#
# BUDGET: 12000 iters @ 2048 envs = 1.33x every previous arm's 9000, and ~3.6 h per
# run at the measured ~1.09 s/iter (queue_overnight, serial). 7 runs ~ 25 h. The
# official runner cfg is max_iterations=50000, so this is still 1/4 of the budget
# the released policy had — the checkpoint sweep (which found our best historical
# deliverable at ck7000, not at the end) stays the tool for reading these runs.
#
# Every arm gets: smoke + switch verification, train, ONNX export, P64 vx=0.2/0.0
# evals, a NEW zero-command standstill probe (gait_forensics: resets / tilt /
# clearance), gait forensics and the acceptance table — all inside run_arm_once.sh.
#
# Usage: nohup setsid bash scripts/queue_family_b.sh > logs/queue_family_b.out 2>&1 &
set -u
cd "$(dirname "$0")/.." || exit 1
OUT=logs/queue_family_b
mkdir -p "$OUT"
export OUT
ITERS=${ITERS:-12000}
export ITERS

log() { printf '[%(%F %T)T] %s\n' -1 "$*" | tee -a "$OUT/status.txt" >&2; }
gpu_free_mib() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -dc '0-9'; }

# ── gate: the checkpoint sweep owns the GPU before us ────────────────────────
waited=0
while :; do
  sweep=$(grep -c "SWEEP COMPLETE" logs/queue_sweep/status.txt 2>/dev/null || true)
  busy=$(pgrep -f "mjlab_microduck[.]train_cli" | wc -l)
  [ "${sweep:-0}" -gt 0 ] && [ "$busy" -eq 0 ] && { log "sweep done; taking the GPU"; break; }
  [ $((waited % 600)) -eq 0 ] && log "waiting: sweep_marker=${sweep:-0} training=${busy} (${waited}s)"
  sleep 60; waited=$((waited + 60))
done
waited=0
while :; do
  free=$(gpu_free_mib || true)
  if [ -n "$free" ] && [ "$free" -ge 9000 ]; then log "GPU ready: ${free} MiB free"; break; fi
  [ $((waited % 300)) -eq 0 ] && log "waiting for GPU (free=${free:-?})"
  sleep 30; waited=$((waited + 30))
done

# ── the recipe: OFFICIAL verbatim + the head block, nothing else ─────────────
# Identical to queue_official_align.sh's ALIGN_OFF, so these arms are the same
# family as align_full / align_fwd and can be pooled with them.
ALIGN_OFF='MICRODUCK_STANCE_HYGIENE=0 MICRODUCK_STANCE_HIP_ROLL=0 MICRODUCK_PER_JOINT_POSTURE=0'
ALIGN_OFF="$ALIGN_OFF MICRODUCK_AIR_TIME_GATE=0 MICRODUCK_HEADING_HOLD=0 MICRODUCK_YAW_HOLD_ZERO_CMD=0"
ALIGN_OFF="$ALIGN_OFF MICRODUCK_ABDUCTION=0 MICRODUCK_YAW_DC=0 MICRODUCK_YAW_SYMMETRY=0"
ALIGN_OFF="$ALIGN_OFF MICRODUCK_TURN_FIRST=0 MICRODUCK_SYMMETRY=0 MICRODUCK_HIP_ROLL_STD=0.05"
ALIGN_OFF="$ALIGN_OFF MICRODUCK_POSE_ZERO_CMD_PROB=0.0"
HEAD_OFFICIAL="$ALIGN_OFF MICRODUCK_HEAD_STABILITY=1"

W_F="Episode_Reward/flight Metrics/air_time_mean Metrics/yaw_dc_abs"
W_H="Episode_Reward/head_motion Metrics/yaw_dc_abs"

# B1: the primary arm — flight penalty on the good family, 3 seeds.
for s in 42 43 44; do
  bash scripts/run_arm_once.sh famB_head_flight "$s" \
    "$HEAD_OFFICIAL MICRODUCK_FLIGHT_PENALTY=1" "$W_F" || log "famB_head_flight s$s FAILED"
done

# B2: the control — the same recipe without the flight penalty.
for s in 43 44; do
  bash scripts/run_arm_once.sh famB_head "$s" "$HEAD_OFFICIAL" "$W_H" || log "famB_head s$s FAILED"
done

# B3: the speed-cap arm (phase-transition test), one variable away from B1.
bash scripts/run_arm_once.sh famB_head_flight_slow 42 \
  "$HEAD_OFFICIAL MICRODUCK_FLIGHT_PENALTY=1 MICRODUCK_CMD_LIN_MAX=0.15" \
  "$W_F" || log "famB_head_flight_slow FAILED"

# B4: conditional — see the header. Cancel it if ours_yawdc failed its bar.
bash scripts/run_arm_once.sh famB_head_flight_yawdc 42 \
  "$HEAD_OFFICIAL MICRODUCK_FLIGHT_PENALTY=1 MICRODUCK_YAW_DC=1" \
  "$W_F Episode_Reward/yaw_dc" || log "famB_head_flight_yawdc FAILED"

log "FAMILY B QUEUE COMPLETE; results: $OUT/results.tsv"
log "read it with: uv run --with pyserial python scripts/report_queue_arms.py  (add the arms to ARMS first)"
