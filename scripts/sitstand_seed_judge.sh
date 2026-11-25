#!/usr/bin/env bash
# =============================================================================
# Judge the FINAL sitstand checkpoint on a DISTRIBUTION, not on one lucky draw.
#
# Why this exists (2026-09-16): the queue evaluates the sitstand arm ONCE, with
# `eval_sitstand_head.py` unseeded (queue_round2.sh line ~191 has no --seed), and
# sitstand DR is NOT gated on play=True. So the Q2 verdict the finalizer writes is
# ONE sample of a random draw. Measured spread on the same artifacts:
#
#     ck3000 seed0 z_min 0.059  seed1 0.058  seed2 0.104  seed3 0.104   -> 2/4
#     ck4000 4/4      ck5000 4/4      ck6000 3/4
#
# i.e. "it sits" and "it cannot sit" are both reachable from ck3000/ck6000
# depending on the draw. A PASS/FAIL read off one draw is therefore not a verdict.
# This watcher waits for the arm to finish, then runs N seeds and reports the
# SUCCESS RATE, and only calls it PASS if EVERY draw clears the bar.
#
# The running queue scripts are NOT touched (bash reads a running script by byte
# offset; editing one mid-flight can make it execute garbage). This is a separate
# process that only READS the run directory -- the same pattern as
# early_tier_judge.sh.
#
# Usage:
#   setsid nohup bash scripts/sitstand_seed_judge.sh > logs/sitstand_seed_judge.out 2>&1 < /dev/null &
# =============================================================================
set -u
cd "$(dirname "$0")/.." || exit 1

export UV_CACHE_DIR=/tmp/uv-cache
export WANDB_MODE=offline

SEEDS="${SEEDS:-0 1 2 3 4 5 6 7}"
TARGET_ITERS="${TARGET_ITERS:-8000}"
OUT=logs/sit_seed_sweep_final
LOG=logs/sitstand_seed_judge.out
MIN_FREE_MIB="${MIN_FREE_MIB:-1200}"
MAX_WAIT_S=$((8 * 3600))

log() { echo "[$(date '+%F %T')] $*"; }
free_mib() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }

# ── locate the sitstand arm's run directory, DATA-DRIVEN ─────────────────────
# Not by hardcoded timestamp, and not by "the newest dir" alone: the arm is
# identified by its own recorded git diff carrying the head-ground fix, which is
# the same evidence standard the rest of this repo uses for "which config trained
# this checkpoint".
find_run_dir() {
  local best="" best_t=0 d t
  for d in logs/rsl_rl/microduck_sitstand/*/; do
    [ -d "$d" ] || continue
    grep -qs "posture_head_ground_penalty" "$d"git/*.diff 2>/dev/null || continue
    [ -f "$d/model_${TARGET_ITERS}.pt" ] || continue
    t=$(stat -c %Y "$d/model_${TARGET_ITERS}.pt")
    if [ "$t" -gt "$best_t" ]; then best_t=$t; best=$d; fi
  done
  printf '%s' "$best"
}

log "waiting for model_${TARGET_ITERS}.pt in a head-ground-fix sitstand run (max ${MAX_WAIT_S}s)"
waited=0
RUN_DIR=""
while :; do
  RUN_DIR=$(find_run_dir)
  [ -n "$RUN_DIR" ] && break
  if [ "$waited" -ge "$MAX_WAIT_S" ]; then
    log "gave up after ${MAX_WAIT_S}s; no model_${TARGET_ITERS}.pt found"
    exit 1
  fi
  [ $((waited % 1800)) -eq 0 ] && log "  still waiting (${waited}s); newest ckpt: $(ls -t logs/rsl_rl/microduck_sitstand/*/model_*.pt 2>/dev/null | head -1)"
  sleep 60; waited=$((waited + 60))
done
log "run_dir = $RUN_DIR"

# Let the trainer finish writing the ckpt's sibling ONNX (the runner re-exports it
# at every save_interval). Wait until the ONNX is at least as new as the ckpt.
ONNX=""
for _ in $(seq 1 30); do
  ONNX=$(ls -t "$RUN_DIR"*.onnx 2>/dev/null | head -1)
  if [ -n "$ONNX" ] && [ "$(stat -c %Y "$ONNX")" -ge "$(stat -c %Y "$RUN_DIR/model_${TARGET_ITERS}.pt")" ]; then
    break
  fi
  log "  waiting for the ONNX to catch up with model_${TARGET_ITERS}.pt"
  sleep 20
done

# Fallback: export it ourselves via the MANDATORY path (bakes the obs normalizer).
if [ -z "$ONNX" ] || [ "$(stat -c %Y "$ONNX")" -lt "$(stat -c %Y "$RUN_DIR/model_${TARGET_ITERS}.pt")" ]; then
  ONNX="/tmp/sit_final_${TARGET_ITERS}.onnx"
  log "ONNX stale/absent -> exporting with scripts/export.py (bakes normalizer)"
  .venv/bin/python scripts/export.py Mjlab-SitStand-Flat-MicroDuck \
      --checkpoint-file "$RUN_DIR/model_${TARGET_ITERS}.pt" --onnx-file "$ONNX" \
      > logs/sit_final_export.log 2>&1
  [ -f "$ONNX" ] || { log "EXPORT FAILED (see logs/sit_final_export.log)"; exit 1; }
fi
log "onnx = $ONNX  ($(sha256sum "$ONNX" | cut -c1-16)…)"

# ── N seeded draws ───────────────────────────────────────────────────────────
log "running ${SEEDS} seeded draws -> $OUT"
for _ in $(seq 1 60); do
  f=$(free_mib); [ "$f" -ge "$MIN_FREE_MIB" ] && break
  log "  waiting for ${MIN_FREE_MIB} MiB free (have ${f})"; sleep 60
done
OUT="$OUT" ARMS="${ONNX}:final_ck${TARGET_ITERS}" SEEDS="$SEEDS" \
  MIN_FREE_MIB="$MIN_FREE_MIB" bash scripts/sit_seed_sweep.sh >> "$LOG" 2>&1

# ── verdict: EVERY draw must clear the bar ───────────────────────────────────
log "==================== Q2 FINAL VERDICT ===================="
.venv/bin/python - "$OUT/summary.tsv" "$TARGET_ITERS" <<'PY'
import sys, statistics as st
from pathlib import Path
p = Path(sys.argv[1]); iters = sys.argv[2]
if not p.exists():
    print("  (no summary.tsv — the sweep did not produce results)"); raise SystemExit(0)
rows = [l.split("\t") for l in p.read_text().splitlines() if l.strip()]
minz = [float(r[3]) for r in rows if r[2].endswith("_MINZ")]
stand = [r for r in rows if r[2].endswith("_STAND")]
headf = [float(r[5]) for r in stand]
headz = [float(r[6]) for r in stand]
tilt  = [float(r[4]) for r in stand]
z     = [float(r[3]) for r in stand]
n = len(minz); ok = sum(1 for v in minz if v <= 0.065)
print(f"  ck{iters}:  {n} seeded draws")
print(f"  SIT  reach z<=0.065 : {ok}/{n}   values {sorted(minz)}")
print(f"  STAND head-ground F : min {min(headf):.3f}  max {max(headf):.3f} N   (bar < 0.5)")
print(f"  STAND head height   : min {min(headz):.1f}  max {max(headz):.1f} mm  (bar > 140)")
print(f"  STAND tilt          : min {min(tilt):.1f}  max {max(tilt):.1f} deg (bar < 20)")
print(f"  STAND z             : min {min(z):.3f}  max {max(z):.3f} m   (bar >= 0.10)")
sit_pass   = ok == n
head_pass  = max(headf) < 0.5 and min(headz) > 140.0 and max(tilt) < 20.0 and min(z) >= 0.10
print()
print(f"  Q2 头三脚架修复 : {'PASS' if head_pass else 'FAIL'}  (every draw)")
print(f"  Q2 能坐下       : {'PASS' if sit_pass else 'FAIL'}  ({n-ok}/{n} draws did NOT sit)")
print(f"  Q2 整体         : {'PASS' if (sit_pass and head_pass) else 'NOT PASSING'}")
if not sit_pass:
    print("  ⚠️  The sit half is draw-dependent. This is the open item; do not report")
    print("      the head fix as 'Q2 done' while this line says FAIL.")
PY
log "summary: $OUT/summary.tsv"
log "==================== DONE ===================="
