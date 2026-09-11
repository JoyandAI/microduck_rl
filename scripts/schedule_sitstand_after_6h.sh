#!/bin/bash
# 定时器：6 小时后启动 sitstand 训练。
#
# 由用户 2026-09-10 18:45 要求：「如果这个模型训练起来了，那就建立一个定时器，
# 6h 后训练坐下起立模型」。
#
# 设计要点（为什么不是纯粹的 sleep 6h 后直接开）：
#   velocity 训练的 ETA 是 ~6h27m（18:43 启动 → 约 01:11 结束），比 6h 的触发点
#   （00:43）更晚。本机只有 12 GB 显存，且 sitstand 比 velocity 重得多
#   （nconmax=200 + solver 30/50 vs 35 + 10/20），两个同时跑必然 OOM。
#   所以顺序是：等到 6h → 再等 velocity 进程退出 → 才启动 sitstand。
#
# 日志：logs/scheduler_sitstand.log（调度器自身）、train_sitstand_hd1910.log（训练）

set -u
cd /home/joyandai/microduck_rl

export UV_CACHE_DIR=/tmp/uv-cache
export WANDB_MODE=offline          # 本机 wandb 未登录，既有做法是离线 run

SCHED_LOG=logs/scheduler_sitstand.log
TRAIN_LOG=train_sitstand_hd1910.log
mkdir -p logs

log() { echo "[$(date '+%F %T')] $*" >> "$SCHED_LOG"; }

log "=============================================================="
log "scheduler started (pid $$). Plan: sleep 6h -> wait for velocity -> start sitstand"
log "velocity run start was 18:43:48, its ETA was 6h27m (~01:11)"

# ── 1) 等 6 小时 ────────────────────────────────────────────────────────────
sleep 21600
log "6h elapsed (trigger point reached)."

# ── 2) 等 velocity 训练退出（最多再等 12h）──────────────────────────────────
waited=0
while pgrep -f "train Mjlab-Velocity" > /dev/null 2>&1; do
  if [ "$waited" -ge 43200 ]; then
    log "WARNING: velocity still running after 12h extra wait; proceeding anyway"
    break
  fi
  sleep 30
  waited=$((waited + 30))
  if [ $((waited % 600)) -eq 0 ]; then
    log "still waiting for velocity to finish (${waited}s)..."
  fi
done
log "velocity training is gone (waited ${waited}s extra)."

# ── 3) 给显存释放留点时间 ───────────────────────────────────────────────────
sleep 60
FREE_MIB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1)
log "GPU free before sitstand: ${FREE_MIB:-unknown} MiB"

# ── 4) 启动 sitstand 训练 ───────────────────────────────────────────────────
# 1024 envs（不是 2048）：sitstand 的 nconmax=200 + solver 30/50 比 velocity 重得多，
# 12 GB 卡上 2048 有 OOM 风险，而 1024 对该任务足够收敛。
log "starting sitstand training: Mjlab-SitStand-Flat-MicroDuck num-envs=1024 iters=15000"
log "  (this run includes the new head_pose_bias droop penalty, scaled x1.032 by head mass)"
nohup uv run train Mjlab-SitStand-Flat-MicroDuck \
  --env.scene.num-envs 1024 \
  --agent.max_iterations 15000 \
  >> "$TRAIN_LOG" 2>&1
rc=$?
log "sitstand training exited with code $rc"

# ── 5) 报告产物 ─────────────────────────────────────────────────────────────
LATEST=$(ls -1dt logs/rsl_rl/microduck_sitstand/*/ 2>/dev/null | head -1)
log "latest sitstand run dir: ${LATEST:-none}"
if [ -n "${LATEST:-}" ]; then
  log "checkpoints: $(ls -1 "${LATEST}"model_*.pt 2>/dev/null | tail -3 | tr '\n' ' ')"
fi
log "ALL DONE"
log "=============================================================="
echo "SCHEDULER_DONE" > logs/scheduler_sitstand.done
