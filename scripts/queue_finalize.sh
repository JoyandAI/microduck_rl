#!/usr/bin/env bash
# =============================================================================
# Close the loop: wait for every training job to finish, then produce the verdict.
#
# Why this exists. The queue's last arm is not expected to finish until ~09:00,
# which is far longer than an interactive session stays open. Without this
# script the verdict exists only as a command somebody has to remember to run,
# and the acceptance table in docs/q1_q2_root_cause_2026-09-16.md keeps its
# `<!--QUEUE_RESULTS-->` placeholder forever.
#
# It waits on BOTH conditions, because either alone is wrong:
#   * no `train_cli Mjlab` process, AND
#   * no queue launcher process (queue_round2*.sh)
# A launcher between arms has no training process yet but will start one.
#
# Then it runs the three judgements in order and writes them into the document:
#   1. `--sweep`  rank EVERY checkpoint of every walk arm (the official ships a
#      selected checkpoint; within-run spread is larger than between-arm spread);
#   2. `--steer`  does a yaw command actually produce yaw? (the user's second
#      requirement: "turn when commanded, straight otherwise" is a CONJUNCTION);
#   3. the acceptance table vs the reference rows.
#
# Usage:
#   setsid nohup bash scripts/queue_finalize.sh > logs/queue_finalize.out 2>&1 < /dev/null &
# =============================================================================
set -u
cd "$(dirname "$0")/.." || exit 1
export UV_CACHE_DIR=/tmp/uv-cache
export WANDB_MODE=offline

OUT=logs/queue_round2
DOC=docs/q1_q2_root_cause_2026-09-16.md
LOG="$OUT/final_verdict.txt"
MAX_WAIT_S=$((24 * 3600))

log() { echo "[$(date '+%F %T')] $*"; }

log "==================================================================="
log "finalizer armed. It will wait for ALL training + launchers to stop,"
log "then sweep every checkpoint, run the steering acceptance, and write"
log "the verdict into $DOC"
log "==================================================================="

waited=0
while :; do
  ntrain=$(pgrep -f "train_cli Mjlab" | wc -l)
  # Match the launchers' ACTUAL command lines ("bash scripts/queue_round2*.sh").
  # A bare `pgrep -f queue_round2` also matches any shell whose command line
  # merely CONTAINS that text — including the one that launched this script —
  # and "subtract 1 defensively" is not a fix: it under-counts whenever the
  # match is absent, which makes the finalizer declare "all quiet" one launcher
  # too early and write a verdict on a half-finished queue.
  nqueue=$(pgrep -f "bash scripts/queue_round2" | wc -l)
  if [ "$ntrain" -eq 0 ] && [ "$nqueue" -eq 0 ]; then
    log "all quiet: 0 training processes, 0 launchers (waited ${waited}s)"
    break
  fi
  [ $((waited % 1800)) -eq 0 ] && log "waiting: ${ntrain} training, ${nqueue} launchers (${waited}s)"
  if [ "$waited" -ge "$MAX_WAIT_S" ]; then
    log "gave up waiting after ${MAX_WAIT_S}s — writing the verdict on what exists"
    break
  fi
  sleep 60; waited=$((waited + 60))
done

log "results.tsv:"
cat "$OUT/results.tsv" 2>/dev/null | sed 's/^/   /' || log "   (missing)"

log "-- step 1/3: checkpoint sweep (every checkpoint of every walk arm)"
.venv/bin/python scripts/queue_round2_report.py --sweep \
    --runs walk_sym,walk_yawdc,turnfirst_yawdc,turn_first 2>&1 | tee "$LOG"

log "-- step 2/3: steering acceptance + acceptance table"
.venv/bin/python scripts/queue_round2_report.py --steer 2>&1 | tee -a "$LOG"

log "-- step 3/3: writing the verdict into $DOC"
.venv/bin/python - "$DOC" "$LOG" "$OUT/results.tsv" <<'PY'
import sys, pathlib, datetime

doc_path, log_path, tsv_path = (pathlib.Path(p) for p in sys.argv[1:4])
doc = doc_path.read_text()
marker = "结果与判定：<!--QUEUE_RESULTS-->"
if marker not in doc:
    print("[!] placeholder not found; leaving the document untouched")
    raise SystemExit(0)

verdict = log_path.read_text() if log_path.exists() else "(no verdict log)"
tsv = tsv_path.read_text() if tsv_path.exists() else "(no results.tsv)"
stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

section = f"""结果与判定（由 `scripts/queue_finalize.sh` 于 {stamp} 自动写入）：

**每个臂的落点**（`logs/queue_round2/results.tsv`）：

```
{tsv.strip()}
```

**判定输出**（完整日志 `{log_path}`）：

```
{verdict.rstrip()}
```

> 读法提醒：`--sweep` 的排名按"**先满足 |yawDC| 与转弯半径，再比路径效率**"排序，
> 并且**拒绝步速低于指令 30% 的检查点** —— "因为几乎不走所以走得很直"不是通过。
> `--steer` 是用户要求的第二个条件：**给出左右指令时必须真的转**。
> 两者同时通过才是可交付的策略；只过一半（例如 `ck7000`）必须明确标注为不可交付。
"""
doc_path.write_text(doc.replace(marker, section))
print(f"[i] verdict written into {doc_path}")
PY

log "==================================================================="
log "DONE. Verdict: $LOG ; document: $DOC"
log "==================================================================="
