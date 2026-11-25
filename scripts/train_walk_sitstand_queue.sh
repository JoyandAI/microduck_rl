#!/usr/bin/env bash
# =============================================================================
# 队列：走路(velocity) + 坐下起立(sitstand) —— 2026-09-15 夜（HD-1910 / 7.5 V BAM）
#
# 用户要求（2026-09-15）：所有测试/评估做完、数据没问题之后，重新安排队列训练
#   「走路」和「坐下起立」两个模型；**走路模型不许在静止(零命令)状态下原地踏步**；
#   把相关参数同步到 rl 里；最后开始训练。
#
# 与 09-14 的 train_skill_queue.sh 的差别（这是"重新安排"的部分）：
#   1. 只跑这两项（原来的 10 技能队列是旧排法；走路/坐起立是当前主线）；
#   2. **两项并发**（彼此独立、不共享 checkpoint）。本机 12 GB 显存：
#      velocity@2048 ≈2.6 GB、sitstand@1024 ≈3–4 GB ⇒ 并发安全；
#      墙钟 ≈ max(走路 3.5 h, 坐起立 2.5 h) 而不是两者相加；
#   3. **参数冻结守卫**：开跑前快照 vendor/bam/.../hd1910 全部参数并记 sha256 指纹，
#      跑完复验。09-10 的事故是两个策略活在两套执行器物理里而无人察觉
#      （见 docs/policy_failure_analysis.md）——现在中途被换会被标记 CONTAMINATED；
#   4. 训完**自动评测**，其中走路的第一项就是"零命令站定"（检测原地踏步）：
#      gait_forensics --lin-vel-x 0 --ang-vel-z 0（冻结指令、去掉推力）
#      + 0.2 m/s 步态 + 直行跟踪；坐起立用 eval_sitstand_head --episode-length-s 60
#      （文档记录的协议 bug 修复后的口径：12 s episode 会在最后一段发生超时重置）。
#
# 走路"不原地踏步"的机制（已在线于 microduck_velocity_env_cfg.py，本次沿用默认）：
#   * stillness_zero_cmd  +0.8   （命令<0.01 时，正奖励随躯干速度递减）
#   * no_stepping         −0.4   （命令≈0 时罚脚离地）
#   * standing_leg_vel    −0.02  （命令≈0 时罚腿关节速度平方和，稠密梯度）
#   三者由 stance_tax_weight 课程在前 300 迭代从 0 爬到目标权重，
#   并由 standing_envs 课程把零命令环境占比从 2% 爬到 25%（≈2000 迭代）。
#   **全部按命令门控 ⇒ 对行走步态零影响。**
#
# 用法：
#   setsid nohup bash scripts/train_walk_sitstand_queue.sh \
#       > logs/queue_walk_sitstand.out 2>&1 < /dev/null &
# 看进度：
#   cat logs/queue_walk_sitstand_status.txt ; tail -f logs/queue_walk_sitstand.out
# =============================================================================
set -u
cd "$(dirname "$0")/.." || exit 1

export WANDB_MODE=offline
export MICRODUCK_TASK_NAMED_RUNS=1      # run 目录按任务名，不用时间戳
export UV_CACHE_DIR=/tmp/uv-cache

OUT=logs/queue_walk_sitstand
STATUS=logs/queue_walk_sitstand_status.txt
RESULTS="$OUT/results.tsv"
PARAMS_DIR=vendor/bam/bam/params/hd1910
SMOKE_ITERS=5
BUDGET_SLACK_S=$((2 * 3600))            # 预计计算时间 + 2h
GPU_WAIT_MAX_S=$((12 * 3600))
CONCURRENCY=${CONCURRENCY:-2}
ONLY="${ONLY:-}"                        # 逗号分隔：只跑这些 run（补跑用）

mkdir -p "$OUT" logs
log() { echo "[$(date '+%F %T')] $*"; }

fingerprint() {
  .venv/bin/python - <<'PY'
from mjlab_microduck.train_hook import actuator_params_fingerprint
print(actuator_params_fingerprint()[0])
PY
}

# name|task_id|iterations|num_envs|为什么这个迭代数
QUEUE=(
  "walk|Mjlab-Velocity-Flat-MicroDuck|12000|2048|步态：AGENTS 预算上限 6000@4096 → 12000@2048；含零命令站定三件套"
  "sitstand|Mjlab-SitStand-Flat-MicroDuck|8000|1024|episodic 坐↔起：cfg 上限 15000，按 AGENTS 简单技巧口径取 8000（最重的 env，用 1024）"
)

FP_BEFORE=$(fingerprint)
STAMP=$(date '+%Y%m%d_%H%M%S')
SNAP="$OUT/params_snapshot_$STAMP"
mkdir -p "$SNAP"
cp -a "$PARAMS_DIR"/. "$SNAP"/

log "=============================================================="
log "walk+sitstand 队列启动 (pid $$)；参数指纹 $FP_BEFORE"
log "参数快照 $SNAP"
log "并发 $CONCURRENCY；run 目录 logs/rsl_rl/<experiment>/<TASK_ID>/"
log "=============================================================="

# ── 等 GPU 空闲 ──────────────────────────────────────────────────────────────
waited=0
while pgrep -f "train_cli Mjlab|train Mjlab" > /dev/null 2>&1; do
  [ $((waited % 600)) -eq 0 ] && log "waiting: another training is running (${waited}s)"
  [ "$waited" -ge "$GPU_WAIT_MAX_S" ] && { log "WARNING: GPU busy ${GPU_WAIT_MAX_S}s — proceeding"; break; }
  sleep 60; waited=$((waited + 60))
done
[ "$waited" -gt 0 ] && log "GPU free (waited ${waited}s)"

[ -f "$RESULTS" ] || printf 'name\ttask\titers\tenvs\trun_dir\trc\tfp_before\tfp_after\telapsed_s\tonnx\teval_zero_cmd\teval_walk02\teval_sitstand\n' > "$RESULTS"

run_budgeted() {  # $1=预算秒 $2=日志 $3..=命令
  local budget="$1" out="$2"; shift 2
  setsid --wait "$@" > "$out" 2>&1 &
  local pid=$! waited=0
  while kill -0 "$pid" 2>/dev/null; do
    sleep 30; waited=$((waited + 30))
    if [ "$waited" -ge "$budget" ]; then
      log "  TIMEOUT: budget ${budget}s → SIGTERM 进程组 $pid"
      kill -TERM -"$pid" 2>/dev/null; sleep 60; kill -KILL -"$pid" 2>/dev/null
      wait "$pid" 2>/dev/null; return 124
    fi
  done
  wait "$pid"
}

run_one() {
  local name="$1" task="$2" iters="$3" envs="$4" why="$5"
  local smoke_log="logs/smoke_${name}.log" train_log="train_${name}.log"
  local started; started=$(date +%s)
  local run_dir="-" onnx="-" trc=0 elapsed=0
  local ev_zero="-" ev_walk="-" ev_sit="-"

  log "── RUN ${name}: ${task} | ${iters} iters @ ${envs} envs"
  log "   why: ${why}"

  MICRODUCK_RUN_DIR_NAME="${task}_smoke" timeout -k 60 1800 \
      .venv/bin/python -m mjlab_microduck.train_cli "$task" \
      --env.scene.num-envs "$envs" --agent.max_iterations "$SMOKE_ITERS" \
      > "$smoke_log" 2>&1
  local smoke_rc=$?
  if [ "$smoke_rc" -ne 0 ] || grep -qE "RuntimeError|Traceback \(most recent" "$smoke_log"; then
    log "   SMOKE FAILED (rc=$smoke_rc) — 跳过 ${name}；见 ${smoke_log}"
    grep -E "RuntimeError|Error|Traceback" "$smoke_log" | tail -3 | sed 's/^/     /'
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$name" "$task" "$iters" "$envs" "smoke-failed" "$smoke_rc" "-" "-" 0 "-" "-" "-" "-" >> "$RESULTS"
    return 1
  fi
  local iter_t
  iter_t=$(grep -o "Iteration time: [0-9.]*s" "$smoke_log" | tail -1 | grep -o "[0-9.]*" || true)
  [ -z "${iter_t:-}" ] && { iter_t=2.0; log "   WARNING: 冒烟日志无 Iteration time，按 ${iter_t}s/iter 估"; }
  local est_s budget_s
  CONC_PENALTY=${CONC_PENALTY:-$CONCURRENCY}   # 冒烟独占；并发时按倍数放宽预算
  est_s=$(python3 -c "print(int(float('${iter_t}')*${iters}*1.15*${CONC_PENALTY}))")
  budget_s=$((est_s + BUDGET_SLACK_S))
  log "   冒烟通过：${iter_t}s/iter（独占）→ 按并发惩罚 ×${CONC_PENALTY} 估 $(python3 -c "print(f'{${est_s}/3600:.1f}')")h，预算 $(python3 -c "print(f'{${budget_s}/3600:.1f}')")h"

  run_budgeted "$budget_s" "$train_log" \
      .venv/bin/python -m mjlab_microduck.train_cli "$task" \
      --env.scene.num-envs "$envs" --agent.max_iterations "$iters" \
      --agent.save_interval 1000
  trc=$?
  elapsed=$(( $(date +%s) - started ))

  run_dir=$(grep -o "Logging experiment in directory: .*" "$train_log" | tail -1 | sed 's/.*directory: //')
  [ -z "${run_dir:-}" ] && run_dir="-"
  onnx=$(ls -t "$run_dir"/*.onnx 2>/dev/null | head -1); [ -z "${onnx:-}" ] && onnx="-"
  log "   训练结束 rc=${trc} 用时 $((elapsed/3600))h$(( (elapsed%3600)/60 ))m"
  log "   run_dir=${run_dir}  onnx=${onnx}"

  # ── 参数冻结复验（09-10 事故的守卫）─────────────────────────────────────────
  local fp_after; fp_after=$(fingerprint)
  if [ "$fp_after" != "$FP_BEFORE" ]; then
    log "   ⚠️ CONTAMINATED：参数指纹在训练期间变了 $FP_BEFORE → $fp_after（本 run 作废）"
  fi

  # ── 评测 ───────────────────────────────────────────────────────────────────
  if [ "$onnx" != "-" ]; then
    if [ "$name" = "walk" ]; then
      # ① 零命令站定 —— 直接回答"静止时会不会原地踏步"
      timeout 1800 .venv/bin/python scripts/gait_forensics.py \
        --onnx "$onnx" --task "$task" --lin-vel-x 0 --lin-vel-y 0 --ang-vel-z 0 \
        --seconds 20 --json-out "$OUT/${name}_zerocmd.json" --tag "zero-cmd stillness" \
        --params-json "$PARAMS_DIR/m5.json" > "$OUT/${name}_zerocmd.log" 2>&1 \
        && ev_zero="$OUT/${name}_zerocmd.json"
      # ② 正常走路步态（0.2 m/s）
      timeout 1800 .venv/bin/python scripts/gait_forensics.py \
        --onnx "$onnx" --task "$task" --lin-vel-x 0.2 --lin-vel-y 0 --ang-vel-z 0 \
        --seconds 20 --json-out "$OUT/${name}_walk02.json" --tag "walk 0.2 m/s" \
        --params-json "$PARAMS_DIR/m5.json" > "$OUT/${name}_walk02.log" 2>&1 \
        && ev_walk="$OUT/${name}_walk02.json"
    else
      timeout 2400 .venv/bin/python scripts/eval_sitstand_head.py \
        --onnx "$onnx" --task "$task" --episode-length-s 60 \
        --params-json "$PARAMS_DIR/m5.json" > "$OUT/${name}_eval.log" 2>&1 \
        && ev_sit="$OUT/${name}_eval.log"
    fi
    log "   评测完成：zero_cmd=${ev_zero} walk02=${ev_walk} sitstand=${ev_sit}"
  else
    log "   ⚠️ 没有 onnx，跳过评测"
  fi

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$name" "$task" "$iters" "$envs" "$run_dir" "$trc" "$FP_BEFORE" "$fp_after" \
    "$elapsed" "$onnx" "$ev_zero" "$ev_walk" "$ev_sit" >> "$RESULTS"

  {
    echo "== ${name} =="
    echo "任务: ${task}   迭代: ${iters} @ ${envs}"
    echo "状态: rc=${trc} $([ "$trc" -eq 124 ] && echo '(超预算被杀)' || echo '正常结束')"
    echo "用时: ${elapsed}s ($(python3 -c "print(f'{${elapsed}/3600:.2f}')")h)"
    echo "run_dir: ${run_dir}"
    echo "参数指纹: ${FP_BEFORE} → ${fp_after} $([ "$fp_after" != "$FP_BEFORE" ] && echo '⚠️ 中途变化!')"
    echo "onnx: ${onnx}"
    echo "评测: zero-cmd=${ev_zero}  walk0.2=${ev_walk}  sitstand=${ev_sit}"
    echo
  } >> "$STATUS"
}

# ── 并发启动（错开 90 s：warp 内核缓存 + 环境构建）──────────────────────────
pids=()
for entry in "${QUEUE[@]}"; do
  IFS='|' read -r name task iters envs why <<< "$entry"
  if [ -n "$ONLY" ] && [[ ",${ONLY}," != *",${name},"* ]]; then continue; fi
  while [ "$(jobs -rp | wc -l)" -ge "$CONCURRENCY" ]; do sleep 20; done
  free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
  while [ "${free_mib:-0}" -lt 3500 ]; do
    log "等待显存（free=${free_mib} MiB）"; sleep 60
    free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
  done
  run_one "$name" "$task" "$iters" "$envs" "$why" &
  pids+=($!)
  sleep 90
done
wait "${pids[@]}" 2>/dev/null || true

log "=============================================================="
log "队列结束。结果表 $RESULTS"
log "指纹 $FP_BEFORE（若某 run 的 fp_after 不同，该 run 作废）"
{
  echo "队列完成: $(date '+%F %T')"
  echo "结果表: ${RESULTS}"
  echo "参数快照: ${SNAP}   指纹: ${FP_BEFORE}"
} >> "$STATUS"
column -s$'\t' -t "$RESULTS" 2>/dev/null || cat "$RESULTS"
