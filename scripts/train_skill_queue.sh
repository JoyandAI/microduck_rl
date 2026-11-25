#!/usr/bin/env bash
# 队列训练：运行时需要的 10 个技能模型，全部用 hd1910 舵机，逐个跑。
#
# 需求（用户 2026-09-11）：「给这 10 个用 hd1910 全部训练一次，一个个训练 8000 步，
# 每 1000 步保存一次；训练时间给宽松一点，在预计计算时间上 +2h」。
#
# 迭代数的依据（详见会话记录）：
#   - episodic 7 个（SitStand/StandUp/GroundPick/Roulade/BallKick×2/RollerCrouch）= 8000。
#     AGENTS.md 的口径是"简单 episodic 技巧 ≈1000 iter @4096 envs" = 等效 2000 @2048，
#     所以 8000 @2048（= 16000 @4096）是它的 8 倍，充足。
#   - 步态 3 个（Velocity / Velocity-Rollers / VelStand）= 12000。AGENTS.md："步态和课程
#     重的恢复类 4000–6000 @4096" = 等效 8000–12000 @2048，取上限。
#   - 各任务 cfg 自带的 max_iterations（velocity 50000 / velstand·groundpick 20000 /
#     standup·sitstand 15000 / ballkick·roulade 10000 / crouch 8000）是"跑满就停"的
#     天花板，不是目标，别照抄。
#
# 显存：本机 12 GB。velocity @2048 实测 2.6 GB；sitstand 一直是最重的
# （nconmax=200 + solver 30/50，历史 run 用 1024），这里保持 1024。
#
# 时间预算：每个任务先跑 5 迭代冒烟 → 从日志读实测 Iteration time → est = t×iters×1.15
# → 超时预算 = est + 2h。超时按**进程组** SIGTERM，最后一份落在 1000 边界上的
# checkpoint 保留（rsl_rl 每 save_interval 存一次）。
#
# run 目录按任务名命名（MICRODUCK_TASK_NAMED_RUNS=1，见 train_cli.py），不是时间戳，
# 所以训练开始前就知道它会在哪：logs/rsl_rl/<experiment_name>/<TASK_ID>/
#
# 用法：
#   setsid nohup bash scripts/train_skill_queue.sh \
#       > logs/queue_skill_hd1910.log 2>&1 < /dev/null &
# 看进度：
#   cat logs/queue_skill_status.txt ; tail -f logs/queue_skill_hd1910.log

set -u
cd "$(dirname "$0")/.." || exit 1

export UV_CACHE_DIR=/tmp/uv-cache
export WANDB_MODE=offline                # 本机 wandb 未登录，沿用既有做法
export MICRODUCK_TASK_NAMED_RUNS=1       # run 目录按任务名，不按时间戳

STATUS=logs/queue_skill_status.txt
RESULTS=logs/queue_skill_results.tsv
SMOKE_ITERS=5
BUDGET_SLACK_S=$((2 * 3600))             # 用户要求：预计计算时间 + 2h
GPU_WAIT_MAX_S=$((12 * 3600))
QUEUE_ONLY="${QUEUE_ONLY:-}"             # 逗号分隔的 run 名：只跑这些（补跑单个任务用）

mkdir -p logs
log() { echo "[$(date '+%F %T')] $*"; }

# name|task_id|iterations|num_envs|为什么这个迭代数
QUEUE=(
  "velocity|Mjlab-Velocity-Flat-MicroDuck|12000|2048|步态：AGENTS 预算上限 6000@4096 → 12000@2048"
  "velstand|Mjlab-VelStand-Flat-MicroDuck|12000|2048|走路+恢复，课程最重的一类"
  "velocity_rollers|Mjlab-Velocity-Flat-MicroDuck-Rollers|12000|2048|步态（被动轮），指令范围更窄"
  "standup|Mjlab-StandUp-Flat-MicroDuck|8000|2048|episodic：四种起始状态课程"
  "sitstand|Mjlab-SitStand-Flat-MicroDuck|8000|1024|episodic；最重的 env，降到 1024 envs"
  "ground_pick|Mjlab-GroundPick-Flat-MicroDuck|8000|2048|episodic：相位命令"
  "roulade|Mjlab-Roulade-Flat-MicroDuck|8000|2048|episodic：一次前滚翻"
  "ball_kick_right|Mjlab-BallKick-Flat-MicroDuck|8000|2048|episodic：右脚踢球"
  "ball_kick_left|Mjlab-BallKick-Flat-MicroDuck-Left|8000|2048|episodic：左脚踢球"
  "roller_crouch|Mjlab-RollerCrouch-Flat-MicroDuck|8000|2048|episodic：= cfg 上限 8000"
)

log "=============================================================="
log "skill queue started (pid $$): ${#QUEUE[@]} runs, hd1910 servos"
log "run dirs: logs/rsl_rl/<experiment>/<TASK_ID>/   (task-named, no timestamps)"
log "=============================================================="

# ── 1) 等 GPU 空闲（当前可能有别的训练在跑）──────────────────────────────────
waited=0
while pgrep -f "train Mjlab" > /dev/null 2>&1; do
  if [ $((waited % 600)) -eq 0 ]; then
    log "waiting: another 'train Mjlab' is running (waited ${waited}s)"
  fi
  if [ "$waited" -ge "$GPU_WAIT_MAX_S" ]; then
    log "WARNING: GPU still busy after ${GPU_WAIT_MAX_S}s — proceeding anyway"
    break
  fi
  sleep 60; waited=$((waited + 60))
done
log "GPU free (waited ${waited}s). Starting queue."
[ -n "$QUEUE_ONLY" ] && log "QUEUE_ONLY=${QUEUE_ONLY} — 只跑这些 run"

# 表头只在文件不存在时写：补跑单个任务不能把之前的成绩抹掉。
[ -f "$RESULTS" ] || printf 'name\ttask\titers\tenvs\trun_dir\trc\telapsed_s\tcheckpoints\tonnx\n' > "$RESULTS"

# ── 2) 带预算地跑一条命令（超时杀整个进程组）─────────────────────────────────
run_budgeted() {  # $1=预算秒  $2=日志文件  $3..=命令
  local budget="$1" out="$2"; shift 2
  setsid --wait "$@" > "$out" 2>&1 &
  local pid=$! waited=0
  while kill -0 "$pid" 2>/dev/null; do
    sleep 30; waited=$((waited + 30))
    if [ "$waited" -ge "$budget" ]; then
      log "  TIMEOUT: budget ${budget}s exceeded -> SIGTERM to process group ${pid}"
      kill -TERM -"$pid" 2>/dev/null
      sleep 60
      kill -KILL -"$pid" 2>/dev/null
      wait "$pid" 2>/dev/null
      return 124
    fi
  done
  wait "$pid"
}

# ── 3) 逐个跑 ────────────────────────────────────────────────────────────────
for entry in "${QUEUE[@]}"; do
  IFS='|' read -r name task iters envs why <<< "$entry"
  if [ -n "$QUEUE_ONLY" ] && [[ ",${QUEUE_ONLY}," != *",${name},"* ]]; then
    continue
  fi
  smoke_log="logs/smoke_${name}.log"
  train_log="train_${name}.log"
  run_dir="-"; onnx_path="-"
  started=$(date +%s)

  {
    echo "当前: ${name}"
    echo "任务: ${task}"
    echo "计划: ${iters} 迭代 @ ${envs} envs，每 1000 存盘"
    echo "开始: $(date '+%F %T')"
    echo "理由: ${why}"
  } > "$STATUS"

  log "──────────────────────────────────────────────────────────────"
  log "RUN ${name}: ${task} | ${iters} iters @ ${envs} envs"
  log "  why: ${why}"

  # 3a) 冒烟（AGENTS.md 铁律：任何长训练前先冒烟）
  log "  smoke: ${SMOKE_ITERS} iters @ ${envs} envs -> ${smoke_log}"
  MICRODUCK_RUN_DIR_NAME="${task}_smoke" timeout -k 60 1800 \
      uv run train "$task" --env.scene.num-envs "$envs" \
      --agent.max_iterations "$SMOKE_ITERS" > "$smoke_log" 2>&1
  smoke_rc=$?
  if [ "$smoke_rc" -ne 0 ] || grep -qE "RuntimeError|Traceback \(most recent" "$smoke_log"; then
    log "  SMOKE FAILED (rc=${smoke_rc}) — skipping this task; see ${smoke_log}"
    grep -E "RuntimeError|Error|Traceback" "$smoke_log" | tail -3 | sed 's/^/    /' | tee -a "$STATUS"
    elapsed=$(( $(date +%s) - started ))
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$name" "$task" "$iters" "$envs" "smoke-failed" "$smoke_rc" "$elapsed" 0 "-" >> "$RESULTS"
    continue
  fi

  # 3b) 用冒烟实测的迭代耗时算预算 = est + 2h
  iter_t=$(grep -o "Iteration time: [0-9.]*s" "$smoke_log" | tail -1 | grep -o "[0-9.]*" || true)
  if [ -z "${iter_t:-}" ]; then
    iter_t=2.0
    log "  WARNING: no 'Iteration time' in the smoke log; assuming ${iter_t}s/iter"
  fi
  est_s=$(python3 -c "print(int(float('${iter_t}') * ${iters} * 1.15))")
  budget_s=$((est_s + BUDGET_SLACK_S))
  log "  measured ${iter_t}s/iter -> est ${est_s}s ($(python3 -c "print(f'{${est_s}/3600:.1f}')")h), budget ${budget_s}s ($(python3 -c "print(f'{${budget_s}/3600:.1f}')")h)"

  {
    echo "预计: ${est_s}s ($(python3 -c "print(f'{${est_s}/3600:.2f}')")h)  = ${iter_t}s/iter × ${iters}"
    echo "预算: ${budget_s}s ($(python3 -c "print(f'{${budget_s}/3600:.2f}')")h)  = 预计 + 2h"
  } >> "$STATUS"

  # 3c) 正式训练
  log "  train -> ${train_log}"
  run_budgeted "$budget_s" "$train_log" \
      uv run train "$task" --env.scene.num-envs "$envs" \
      --agent.max_iterations "$iters" --agent.save_interval 1000
  train_rc=$?

  run_dir=$(grep -o "Logging experiment in directory: .*" "$train_log" | tail -1 | sed 's/.*directory: //')
  [ -z "${run_dir:-}" ] && run_dir="-"
  nckpt=0
  [ -d "$run_dir" ] && nckpt=$(ls "$run_dir"/model_*.pt 2>/dev/null | wc -l)
  last_it=$(ls "$run_dir"/model_*.pt 2>/dev/null | sed 's/.*model_//;s/\.pt//' | sort -n | tail -1)
  elapsed=$(( $(date +%s) - started ))

  log "  done rc=${train_rc} elapsed=$((elapsed/3600))h$(( (elapsed%3600)/60 ))m"
  log "  run_dir=${run_dir}  checkpoints=${nckpt}  last_iter=${last_it:-none}"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$name" "$task" "$iters" "$envs" "$run_dir" "$train_rc" "$elapsed" "$nckpt" "$onnx_path" >> "$RESULTS"

  {
    echo "状态: rc=${train_rc} $([ "$train_rc" -eq 124 ] && echo '(超预算被杀)' || echo '正常结束')"
    echo "耗时: ${elapsed}s ($(python3 -c "print(f'{${elapsed}/3600:.2f}')")h)"
    echo "run_dir: ${run_dir}"
    echo "checkpoints: ${nckpt} 个，最后 ${last_it:-none}"
    echo "结束: $(date '+%F %T')"
  } >> "$STATUS"
done

log "=============================================================="
log "queue finished. results: ${RESULTS}"
log "=============================================================="
{ echo "队列完成: $(date '+%F %T')"; echo "结果表: ${RESULTS}"; } >> "$STATUS"
