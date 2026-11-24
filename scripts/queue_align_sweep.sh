#!/usr/bin/env bash
# Checkpoint sweep for "how straight can it walk" — the KPI the user set on
# 2026-09-17 ("现在就是看能走多直").
#
# WHY A SWEEP AND NOT JUST THE FINAL CHECKPOINT: the repo's own best deliverable
# (`walk_hd1910_7000.onnx`) was a MID-RUN checkpoint, not the last one. Straightness
# is not monotone in training time, so the question "how straight does this recipe
# get" is answered by the best checkpoint of the run, not by model_9000.
#
# Protocol: 64 envs x 20 s, seed 0, DR on, frozen command (AGENTS.md: never quote
# single-env magnitudes; the 64-env protocol is the quotable one). Warmup 2 s,
# `--freeze-pose-cmd` so the head_pose command is the deployment idle state.
#
# Waits for the alignment queue to finish first, so it never competes for the GPU
# with a training run.
#
# Usage: nohup setsid bash scripts/queue_align_sweep.sh > logs/queue_align_sweep.out 2>&1 &
set -u

cd "$(dirname "$0")/.." || exit 1
OUT=logs/queue_align/sweep
QUEUE_STATUS=logs/queue_align/status.txt
mkdir -p "$OUT"

export UV_CACHE_DIR=/tmp/uv-cache
export MPLCONFIGDIR=/tmp/matplotlib

TASK="Mjlab-Velocity-Flat-MicroDuck"
STRIDE=${STRIDE:-1000}          # evaluate every Nth iteration, plus the last
GPU_MIN_FREE=${GPU_MIN_FREE:-9000}

log() { printf '[%(%F %T)T] %s\n' -1 "$*" | tee -a "$OUT/status.txt" >&2; }

gpu_free_mib() {
  nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null |
    head -1 | tr -dc '0-9'
}

wait_for_queue_done() {
  # BOTH conditions, not just the queue's own "complete" marker: a third arm
  # (scripts/queue_align_arm3.sh) trains alongside the two queued arms and writes
  # to the same status file, so the marker alone would let this sweep start while
  # that arm is still producing checkpoints — and it would then miss the later ones.
  local waited=0
  while :; do
    local marked=0 busy=0
    grep -q "queue complete" "$QUEUE_STATUS" 2>/dev/null && marked=1
    pgrep -f "mjlab_microduck[.]train_cli" >/dev/null && busy=1
    if [ "$marked" -eq 1 ] && [ "$busy" -eq 0 ]; then
      log "all training finished (marker + no train_cli) after ${waited}s"
      return 0
    fi
    [ $((waited % 300)) -eq 0 ] && \
      log "waiting: marker=${marked} training_busy=${busy} (${waited}s)"
    sleep 60
    waited=$((waited + 60))
  done
}

wait_for_gpu() {
  local waited=0 free
  while :; do
    free=$(gpu_free_mib || true)
    if [ -n "$free" ] && [ "$free" -ge "$GPU_MIN_FREE" ]; then
      log "GPU ready: ${free} MiB free"
      return 0
    fi
    [ $((waited % 300)) -eq 0 ] && log "waiting for GPU (free=${free:-?}, need ${GPU_MIN_FREE})"
    sleep 30
    waited=$((waited + 30))
  done
}

# Evaluate one ONNX on the straightness protocol and append a TSV row.
# Metrics chosen per AGENTS.md: yaw DC (the robust one), net yaw (wrap-safe, from
# the evaluator), path efficiency WITH its survival context, and the median
# recorded window — never a bare ratio.
eval_one() {
  local label="$1" onnx="$2" json="$OUT/${label}_walk.json"
  .venv/bin/python scripts/eval_straightness.py --onnx "$onnx" --task "$TASK" \
    --lin-vel-x 0.2 --num-envs 64 --seed 0 --seconds 20 --freeze-pose-cmd \
    --json "$json" > "$OUT/${label}_walk.log" 2>&1
  local rc=$?
  if [ $rc -ne 0 ] || [ ! -f "$json" ]; then
    printf '%s\t%s\t-\t-\t-\t-\t-\t-\t-\n' "$label" "eval_rc=$rc" >> "$TSV"
    return 1
  fi
  .venv/bin/python - "$label" "$json" "$TSV" <<'PY'
import json, sys
label, path, tsv = sys.argv[1], sys.argv[2], sys.argv[3]
d = json.load(open(path))
agg = d.get("aggregate", {})

def g(key, sub="mean"):
    v = agg.get(key)
    return v.get(sub) if isinstance(v, dict) else v

row = [
    label,
    f"{g('yaw_rate_bias_deg_s'):+.3f}",
    f"{agg.get('net_yaw_deg__median', float('nan')):+.1f}",
    f"{agg.get('path_efficiency__median', float('nan')):.3f}",
    f"{d.get('median_window_s', float('nan')):.2f}",
    f"{d.get('frac_envs_full_window', float('nan')):.2f}",
    str(d.get("falls")),
    f"{g('speed_along_mps'):.3f}",
    f"{g('tilt_mean_deg'):.2f}",
]
print("\t".join(row), file=open(tsv, "a"))
PY
}

TSV="$OUT/straightness.tsv"
printf 'label\tyaw_dc_deg_s\tnet_yaw_med_deg\tpath_eff_med\tmedian_window_s\tfull_window_frac\tfalls\tspeed_along_mps\ttilt_deg\n' > "$TSV"

# ── marching ("原地踏步") ────────────────────────────────────────────────────
# The acceptance table does NOT cover it (checked 2026-09-17), and the official
# recipe contains no explicit anti-marching term — it relies on the standing-env
# curriculum alone. The repo's chosen metric is FOOT CLEARANCE at zero command
# (marching 19.4 mm vs standing 0.5-4.2 mm; threshold 8 mm, see
# scripts/queue_walk_sitstand_report.py and AGENTS.md). So measure it instead of
# assuming it.
MARCH_TSV="$OUT/marching.tsv"
printf 'label\tzero_cmd_clearance_mm\ttilt_deg\ttrunk_z_mm\tnet_yaw_deg\tfalls\n' > "$MARCH_TSV"

measure_marching() {
  local label="$1" onnx="$2" json="$OUT/${label}_stand_forensics.json"
  .venv/bin/python scripts/gait_forensics.py --onnx "$onnx" --task "$TASK" \
    --lin-vel-x 0.0 --warmup 2 --seconds 20 --head-cmd zero \
    --json-out "$json" > "$OUT/${label}_stand.log" 2>&1
  if [ ! -f "$json" ]; then
    printf '%s\t-\t-\t-\t-\t-\n' "$label" >> "$MARCH_TSV"; return 1
  fi
  .venv/bin/python - "$label" "$json" "$MARCH_TSV" <<'PY'
import json, sys
label, path, tsv = sys.argv[1], sys.argv[2], sys.argv[3]
d = json.load(open(path))
gait = d.get("gait", {})
trunk = d.get("trunk", {})
row = [
    label,
    # key is foot_clearance_body_z_m (METRES); `foot_clearance_mm` does not exist
    f"{(gait.get('foot_clearance_body_z_m') or float('nan')) * 1000:.2f}",
    f"{trunk.get('tilt_mean_deg', float('nan')):.2f}",
    f"{trunk.get('z_mean', float('nan')) * 1000:.1f}",
    f"{d.get('net_yaw_deg', float('nan')):+.1f}",
    str(d.get("falls", d.get("n_resets", "-"))),
]
print("\t".join(row), file=open(tsv, "a"))
PY
}

wait_for_queue_done
wait_for_gpu

for arm in align_full align_official; do
  run="logs/rsl_rl/velocity/$arm"
  [ -d "$run" ] || { log "[$arm] no run dir; skipping"; continue; }
  mapfile -t cks < <(ls "$run"/model_*.pt 2>/dev/null | sort -t_ -k2 -n)
  last=$(printf '%s\n' "${cks[@]}" | tail -1)
  log "[$arm] ${#cks[@]} checkpoints; evaluating every ${STRIDE} iters + the last"
  for ck in "${cks[@]}"; do
    it=$(basename "$ck" .pt | sed 's/model_//')
    # skip the untrained model_0 (a random policy is not a data point)
    if [ "$it" -eq 0 ]; then continue; fi
    if [ "$((it % STRIDE))" -ne 0 ] && [ "$ck" != "$last" ]; then continue; fi
    # idempotent with the live sweep: never evaluate a label twice
    if cut -f1 "$TSV" | grep -qx "${arm}_ck$(printf '%05d' "$it")"; then continue; fi
    label="${arm}_ck$(printf '%05d' "$it")"
    onnx="$OUT/${label}.onnx"
    if [ ! -f "$onnx" ]; then
      .venv/bin/python scripts/export.py "$TASK" --checkpoint-file "$ck" \
        --onnx-file "$onnx" > "$OUT/${label}_export.log" 2>&1 || {
          log "[$label] export failed"; continue; }
    fi
    eval_one "$label" "$onnx" || log "[$label] eval failed"
    # marching is measured on the same checkpoints: it is requirement #2/#3 and
    # the official recipe has no explicit term for it, so it must be observed.
    measure_marching "$label" "$onnx" || log "[$label] marching measurement failed"
    log "[$label] done"
  done
done

# Rank by the KPI. `sort -k3` sorts on net_yaw_med_deg (median |net yaw| proxy);
# lower |yaw DC| first, since that is the number the official reference quotes.
log "=== ranking (straightest first, by |yaw DC|) ==="
tail -n +2 "$TSV" | awk -F'\t' '{v=$2; if (v<0) v=-v; print v"\t"$0}' |
  sort -g | cut -f2- | tee "$OUT/ranking.tsv"

# Threshold is the repo's own (queue_walk_sitstand_report.py: 8 mm); official
# reference is 0.00 mm at zero command, our previous baseline 0.15-0.23 mm, and
# the marching defect 19.4 mm.
log "=== marching at zero command (threshold 8 mm; official 0.00, defect 19.4) ==="
column -t -s$'\t' "$MARCH_TSV" | tee -a "$OUT/status.txt"

# The three checks that decide "aligned with official", printed together so the
# verdict needs no interpretation later.
log "=== verdict table (64 envs, seed 0, frozen command) ==="
.venv/bin/python - "$TSV" "$MARCH_TSV" "$OUT/verdict.txt" <<'PY'
import csv, sys
walk = {r["label"]: r for r in csv.DictReader(open(sys.argv[1]), delimiter="\t")}
march = {r["label"]: r for r in csv.DictReader(open(sys.argv[2]), delimiter="\t")}
REF = {"yaw_dc": 0.23, "path_eff": 0.30, "clearance": 8.0}  # official / repo bars
rows = ["label\tyaw_dc\t|net yaw|\tpath_eff\tclearance_mm\tfalls\tverdict"]
for lbl, w in sorted(walk.items(), key=lambda kv: abs(float(kv[1]["yaw_dc_deg_s"] or 0))):
    m = march.get(lbl, {})
    try:
        straight = abs(float(w["yaw_dc_deg_s"])) < 1.0
        eff = float(w["path_eff_med"]) >= 0.30
        still = float(m.get("zero_cmd_clearance_mm", "nan")) < REF["clearance"]
    except (TypeError, ValueError):
        rows.append(f"{lbl}\t{w['yaw_dc_deg_s']}\t-\t-\t-\t-\tNO DATA"); continue
    ok = straight and eff and still
    rows.append("\t".join([
        lbl, w["yaw_dc_deg_s"], w["net_yaw_med_deg"], w["path_eff_med"],
        m.get("zero_cmd_clearance_mm", "-"), w["falls"],
        "PASS" if ok else "fail:" + ",".join(
            n for n, v in (("yaw", straight), ("eff", eff), ("march", still)) if not v),
    ]))
open(sys.argv[3], "w").write("\n".join(rows) + "\n")
print("\n".join(rows))
PY

log "sweep complete; tables: $TSV / $MARCH_TSV / $OUT/verdict.txt"
