#!/usr/bin/env bash
# =============================================================================
# 收尾：等 walk+sitstand 队列跑完 → 出判定 → 导出交付件 → 生成最终报告
#
#   bash scripts/finalize_walk_sitstand_queue.sh          # 等队列结束（最多 16h）
#   REPEATS=5 ONNX_REPEATS=1 bash scripts/...             # 覆盖评测重复次数
#
# 产物：
#   logs/queue_walk_sitstand/report.json        双判据判定（离地/动作率/tilt/指纹 + 坐起立 z/tilt）
#   walk_hd1910_<iters>.onnx / sitstand_hd1910_<iters>.onnx   （仓库根，交付件，附 .sha256）
#   docs/walk_sitstand_final_report.md          最终报告（判定表 + 校验和 + 机器人侧替换命令）
#
# ⚠️ 本脚本**不改动机器人仓库**（/home/joyandai/microduck）：只打印可复制的替换命令。
# =============================================================================
set -u
cd "$(dirname "$0")/.." || exit 1

OUT=logs/queue_walk_sitstand
RESULTS="$OUT/results.tsv"
REPEATS="${REPEATS:-5}"
ONNX_REPEATS="${ONNX_REPEATS:-1}"
WAIT_MAX_S="${WAIT_MAX_S:-57600}"        # 16h
PY=".venv/bin/python"

log() { echo "[$(date '+%F %T')] $*"; }

# ── 1) 等队列结束（三个标记：DONE_ALL，或 results.tsv 已有两行数据）──────
waited=0
while [ ! -f "$OUT/DONE_ALL" ]; do
  rows=$(($(wc -l < "$RESULTS" 2>/dev/null || echo 1) - 1))
  if [ "$rows" -ge 2 ]; then break; fi
  if pgrep -f "train_walk_sitstand_queue" > /dev/null 2>&1; then
    [ $((waited % 900)) -eq 0 ] && log "等待队列…（${waited}s，已完成 ${rows}/2）"
  else
    log "队列进程已不在，且只完成 ${rows}/2 —— 继续等标记"
  fi
  [ "$waited" -ge "$WAIT_MAX_S" ] && { log "等待超时"; break; }
  sleep 60; waited=$((waited + 60))
done
log "队列结束（等待 ${waited}s）"

# ── 2) 判定 ────────────────────────────────────────────────────────────────
log "跑判定（零命令重复 ${REPEATS} 次）…"
$PY scripts/queue_walk_sitstand_report.py --repeats "$REPEATS" | tee "$OUT/report_stdout.txt"

# ── 3) 导出交付件（仓库根 + sha256）────────────────────────────────────────
declare -A ONNX_FOR ITERS_FOR
while IFS=$'\t' read -r name _task iters _envs _rd _rc _fb _fa _el onnx _a _b _c; do
  [ "$name" = "name" ] && continue
  [ -n "${onnx:-}" ] && [ "$onnx" != "-" ] && ONNX_FOR[$name]="$onnx"
  ITERS_FOR[$name]="$iters"
done < "$RESULTS"

for name in walk sitstand; do
  src="${ONNX_FOR[$name]:-}"
  [ -z "$src" ] && { log "!! ${name}: results.tsv 里没有 onnx 路径"; continue; }
  iters="${ITERS_FOR[$name]:-}"          # run 的**训练迭代数**（onnx 文件名里没有数字）
  dst="${name}_hd1910_${iters:-final}.onnx"
  cp -f "$src" "$dst"
  sha256sum "$dst" > "$dst.sha256"
  log "交付件 $dst  ($(md5sum "$dst" | cut -c1-12) md5, $(stat -c%s "$dst") B)"
done

# ── 4) 最终报告 ────────────────────────────────────────────────────────────
{
  echo "# 走路 + 坐下起立 最终报告（$(date '+%Y-%m-%d %H:%M')）"
  echo
  echo "队列：\`scripts/train_walk_sitstand_queue.sh\`；结果表 \`$RESULTS\`；判定 \`$OUT/report.json\`。"
  echo
  echo "## 判定"
  echo
  echo '```'
  cat "$OUT/report_stdout.txt"
  echo '```'
  echo
  echo "## 交付件"
  echo
  echo "| 文件 | 大小 | sha256 |"
  echo "|---|---:|---|"
  for f in "walk_hd1910_${ITERS_FOR[walk]:-final}.onnx" "sitstand_hd1910_${ITERS_FOR[sitstand]:-final}.onnx"; do
    [ -f "$f" ] || continue
    sha=$([ -f "$f.sha256" ] && cut -d' ' -f1 "$f.sha256" || echo "n/a")
    echo "| \`$f\` | $(stat -c%s "$f") B | \`$sha\` |"
  done
  echo
  echo "## 机器人侧替换（**未执行**，需你确认）"
  echo
  echo '```bash'
  echo "# 1) 备份现网策略"
  echo "cp /home/joyandai/microduck/policies/alpha_walking.onnx{,.bak.\$(date +%F)}"
  echo "cp /home/joyandai/microduck/policies/alpha_sitstand.onnx{,.bak.\$(date +%F)}"
  echo "# 2) 放入新策略（保持运行时约定名，或改配置 walk = .../policies/<新名>.onnx）"
  echo "cp walk_hd1910_${ITERS_FOR[walk]:-final}.onnx /home/joyandai/microduck/policies/alpha_walking.onnx"
  echo "cp sitstand_hd1910_${ITERS_FOR[sitstand]:-final}.onnx /home/joyandai/microduck/policies/alpha_sitstand.onnx"
  echo "# 3) 部署前彩排（CPU、忠实 BAM 环境）"
  echo ".venv/bin/python scripts/infer_policy.py --walking walk_hd1910_${ITERS_FOR[walk]:-final}.onnx"
  echo '```'
} > docs/walk_sitstand_final_report.md
log "最终报告 → docs/walk_sitstand_final_report.md"
log "完成"
