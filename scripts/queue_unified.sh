#!/usr/bin/env bash
# UNIFIED MODEL (2026-09-21) — user: "把现在训练的都停掉，总结合并一下所有的训练参数
# 优化方向，然后再训练一个统一的出来".
#
# Every line below is a direction that was MEASURED to help, and nothing else is on.
# Evidence (all on one protocol: 64 env, 20 s, frozen command, same robot/servo):
#
#  ① 官方配方（12 个偏差全关）
#      ours_flight_s42  : 35/64 会走, 0.026 m/s, 净转角 160.8°, 205°/m
#      famB_head_flight : 60/64 会走, 0.117 m/s, 净转角  57.7°,  21.5°/m   ← 同 seed 配对
#  ② 头块（HEAD_STABILITY=1）
#      官方配方 verbatim 9000it 只有 7/64 会走；+头块 → 24–51/64。也是用户的原始要求。
#  ③ flight 惩罚（FLIGHT_PENALTY=1）
#      flight 0.229 → 0.017–0.043（官方 0.000）；零命令从 62–64 次摔 → 1 次；
#      6 个 run（12偏差族 ×3 + r2 ×3）上一致，站立离地 17–69 mm → 0.3–0.9 mm。
#  ④ 零命令三件套（ZERO_CMD_TERMS=1，新开关）
#      famB 会走也会站，但零命令下 2.15 Hz、脚抬 14.7 mm = 原地踏步（用户指出）。
#      同速度对照（都 ~0.115 m/s）：12 偏差 + flight 的 ours_flight_s42_r2 只有 0.34 mm
#      —— 差别就是 stillness_zero_cmd + no_stepping + yaw_hold_zero_cmd。
#      三项都按 `cmd_speed < 0.01` 门控，走路时恒为 0 ⇒ 对步态零影响（构造保证）。
#      YAW_HOLD_ZERO_CMD=1 是必需的，否则第三项的爬坡权重恒为 0（死项）。
#  ⑤ 预算 12000 it（famB 用的就是 12k；官方 runner cfg 是 50000）
#  ⑥ seed 42（famB 里唯一会走的 seed：60/64 vs s43 的 43/64；s44 未跑完就按用户要求停了）
#
# 不合并（都被实测否掉）：turn_first（转向增益 1.9–5.7% vs 官方 57.6%）、yaw_z_track
# （234° 无改善）、yaw_dc 单用（站立 56 次摔）、feet_max_width 单用（站立 86 mm 乱蹬）、
# hip_roll 收紧/L1（存活 100%→0%）；命令速度上限 0.15 m/s 未验证且会降速，故不加。
#
# 判定（跑完由 run_arm_once.sh 自动产出）：走动 ≥50/64 且速度 ≥0.09 m/s；
# 零命令 clearance <8 mm 且无步态节奏、resets ≤1、tilt <10°；净转角 ≤80°、0 摔。
#
# Usage: nohup setsid bash scripts/queue_unified.sh > logs/queue_unified.out 2>&1 &
set -u
cd "$(dirname "$0")/.." || exit 1
export OUT=logs/queue_unified
mkdir -p "$OUT"
export ITERS=${ITERS:-12000}
export ENVS=${ENVS:-2048}

UNIFIED='MICRODUCK_STANCE_HYGIENE=0 MICRODUCK_STANCE_HIP_ROLL=0 MICRODUCK_PER_JOINT_POSTURE=0'
UNIFIED="$UNIFIED MICRODUCK_AIR_TIME_GATE=0 MICRODUCK_HEADING_HOLD=0 MICRODUCK_YAW_HOLD_ZERO_CMD=1"
UNIFIED="$UNIFIED MICRODUCK_ABDUCTION=0 MICRODUCK_YAW_DC=0 MICRODUCK_YAW_SYMMETRY=0"
UNIFIED="$UNIFIED MICRODUCK_TURN_FIRST=0 MICRODUCK_SYMMETRY=0 MICRODUCK_HIP_ROLL_STD=0.05"
UNIFIED="$UNIFIED MICRODUCK_POSE_ZERO_CMD_PROB=0.0 MICRODUCK_HEAD_STABILITY=1"
UNIFIED="$UNIFIED MICRODUCK_FLIGHT_PENALTY=1 MICRODUCK_ZERO_CMD_TERMS=1"

bash scripts/run_arm_once.sh unified 42 "$UNIFIED" \
  "Episode_Reward/flight Episode_Reward/no_stepping Metrics/yaw_dc_abs" \
  || { printf '[%(%F %T)T] UNIFIED RUN FAILED\n' -1 | tee -a "$OUT/status.txt"; exit 1; }
printf '[%(%F %T)T] UNIFIED RUN COMPLETE; results %s/results.tsv\n' -1 "$OUT" | tee -a "$OUT/status.txt"
