#!/usr/bin/env bash
# Best-checkpoint sweep for the arms of the 2026-09-18/19 queues, driven from the
# RUN DIRECTORIES (not from a queue's exported .onnx).
#
# WHY NOT REUSE THE QUEUES' ONNX: `run_arm_once.sh` resolved the run dir with
# `[ -d "$conventional_path" ] || parse-the-log`, so the overnight re-runs of
# `ours_flight_s42/43/44` (which `MICRODUCK_TASK_NAMED_RUNS=1` had silently
# renamed to `*_r2`) exported the PREVIOUS (routeA) checkpoint instead. Proven by
# matching ONNX `mlp.2.weight` against both runs' `model_8999.pt`:
#     routeA  max|Δ| = 0.000e+00   ← the queue's ONNX was THIS one
#     r2      max|Δ| = 5.733e-01
# The script is fixed now, but those three `_r2` runs (~9 GPU-h) still have no
# evaluation. This sweep exports from the run dirs directly, which also fixes a
# second, older problem: the queues only ever evaluated the FINAL checkpoint, and
# the repo's own best deliverable (`walk_hd1910_7000.onnx`) was a MID-RUN one —
# straightness is not monotone in training time (measured: `ours_head_ck05000`
# scores 109 deg vs 427 deg for the same run's final checkpoint).
#
# Gating: waits for `EXTENSION QUEUE COMPLETE`, then for an idle GPU, so it never
# competes with training (a concurrent run halves the iteration rate and can make
# a queued arm's own budget SIGTERM it).
#
# Usage: nohup setsid bash scripts/queue_ckpt_sweep_arms.sh > logs/queue_sweep.out 2>&1 &
set -u
cd "$(dirname "$0")/.." || exit 1
OUT=${SWEEP_OUT:-logs/queue_sweep}
mkdir -p "$OUT"
TSV="$OUT/straightness.tsv"
STAND_TSV="$OUT/marching.tsv"
STRIDE=${STRIDE:-1500}
TASK="Mjlab-Velocity-Flat-MicroDuck"
GPU_MIN_FREE=${GPU_MIN_FREE:-9000}

export UV_CACHE_DIR=/tmp/uv-cache MPLCONFIGDIR=/tmp/matplotlib WANDB_MODE=offline
log() { printf '[%(%F %T)T] %s\n' -1 "$*" | tee -a "$OUT/status.txt" >&2; }
gpu_free_mib() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -dc '0-9'; }

# SKIP_WAIT=1: the caller already owns the GPU (used by queue_family_b2.sh to sweep
# the family-B arms right after it trains them, without re-waiting on a marker).
if [ "${SKIP_WAIT:-0}" != "1" ]; then
# ── gate 1: the training queue is done (it owns the GPU) ─────────────────────
# 2026-09-20: this used to wait for the EXTENSION queue's marker. That queue was
# cancelled — all three of its arms are our 12-deviation recipe, which the family-B
# analysis showed is the worst configuration we have — so waiting for its marker
# would block this sweep forever. The overnight queue is the last training job
# before us now.
waited=0
while :; do
  done_marker=$(grep -c "OVERNIGHT QUEUE COMPLETE" logs/queue_overnight/status.txt 2>/dev/null || true)
  busy=$(pgrep -f "mjlab_microduck[.]train_cli" | wc -l)
  [ "${done_marker:-0}" -gt 0 ] && [ "$busy" -eq 0 ] && { log "queues done (overnight marker present)"; break; }
  [ $((waited % 600)) -eq 0 ] && log "waiting: overnight_marker=${done_marker:-0} training=${busy} (${waited}s)"
  sleep 60; waited=$((waited + 60))
done

# ── gate 2: GPU actually idle ───────────────────────────────────────────────
waited=0
while :; do
  free=$(gpu_free_mib || true)
  if [ -n "$free" ] && [ "$free" -ge "$GPU_MIN_FREE" ]; then log "GPU ready: ${free} MiB free"; break; fi
  [ $((waited % 300)) -eq 0 ] && log "waiting for GPU (free=${free:-?}, need ${GPU_MIN_FREE})"
  sleep 30; waited=$((waited + 30))
done

fi   # end SKIP_WAIT

printf 'label\tarm\tckpt\tyaw_dc_deg_s\tnet_yaw_med_deg\tpath_eff_med\tmedian_window_s\tfull_window_frac\tfalls\tspeed_along_mps\ttilt_deg\n' > "$TSV"

# Arm list, trimmed 2026-09-20 to what the family-B decision needs. The deviant-recipe
# single-seed probes (ours_stance / ours_ztrack / ours_flight_stance / ours_yawdc) are
# dropped: family B supersedes that whole family, and their final checkpoints are
# already evaluated by the queue itself. Kept: the 3-seed baseline, the flight arm in
# BOTH runs (the `_r2` runs were never evaluated — see the provenance bug in
# docs/queue_results_2026-09-20.md §5), and the alignment arms, which are the current
# straightness leaders and the family B is built on. Re-evaluating them in ONE batch is
# what makes their numbers comparable at all (cross-date drift is ~10 deg).
# SWEEP_ARMS="a b c" overrides the list (space separated run-dir names).
if [ -n "${SWEEP_ARMS:-}" ]; then
  read -r -a ARMS <<< "$SWEEP_ARMS"
else
ARMS=(
  ours_head ours_head_s43 ours_head_s44
  ours_flight_s42 ours_flight_s43 ours_flight_s44
  ours_flight_s42_r2 ours_flight_s43_r2 ours_flight_s44_r2
  align_fwd align_full align_official
)
fi

eval_one() {  # label onnx
  local label="$1" onnx="$2" json="$OUT/${label}_walk.json"
  .venv/bin/python scripts/eval_straightness.py --onnx "$onnx" --task "$TASK" \
    --lin-vel-x 0.2 --num-envs 64 --seed 0 --seconds 20 --freeze-pose-cmd \
    --json "$json" > "$OUT/${label}_walk.log" 2>&1
  if [ ! -f "$json" ]; then
    printf '%s\t-\t-\t-\t-\t-\t-\t-\t-\t-\n' "$label" "-" >> "$TSV"; return 1
  fi
  .venv/bin/python - "$label" "$json" "$TSV" <<'PY'
import json, sys
label, path, tsv = sys.argv[1], sys.argv[2], sys.argv[3]
d = json.load(open(path)); agg = d.get("aggregate", {})
def g(key, sub="mean"):
    v = agg.get(key); return v.get(sub) if isinstance(v, dict) else v
arm, _, ck = label.rpartition("_ck")
print("\t".join([
    label, arm, ck,
    f"{g('yaw_rate_bias_deg_s'):+.3f}",
    f"{agg.get('net_yaw_deg__median', float('nan')):+.1f}",
    f"{agg.get('path_efficiency__median', float('nan')):.3f}",
    f"{d.get('median_window_s', float('nan')):.2f}",
    f"{d.get('frac_envs_full_window', float('nan')):.2f}",
    str(d.get("falls")), f"{g('speed_along_mps'):.3f}", f"{g('tilt_mean_deg'):.2f}",
]), file=open(tsv, "a"))
PY
}

# ── standstill probe: requirement #2/#3 ("不许原地踏步 / 站定也不许踏步") ──────
# The queues never measured this: their `_vx0.0.json` came back with
# `n_envs_insufficient_data = 64` (every env dropped), and `run_arm_once.sh` has no
# stand probe at all, so the whole family's zero-command behaviour is unmeasured.
# The repo's validated metric is FOOT CLEARANCE at zero command (marching 19.4 mm
# vs standing 0.5-4.2 mm; bar 8 mm) — a nan/absent clearance means "no swing", i.e.
# it is NOT marching. Also record the zero-command NET YAW, because "spins in place"
# is the other failure mode and it hides behind a perfect linear speed
# (measured: align_full 306 deg at zero command vs align_fwd 1.3 deg, same config).
measure_stand() {  # label onnx
  local label="$1" onnx="$2" json="$OUT/${label}_stand.json"
  .venv/bin/python scripts/gait_forensics.py --onnx "$onnx" --task "$TASK" \
    --lin-vel-x 0.0 --warmup 2 --seconds 20 --head-cmd zero \
    --json-out "$json" > "$OUT/${label}_stand.log" 2>&1
  if [ ! -f "$json" ]; then
    printf '%s\t-\t-\t-\t-\t-\n' "$label" >> "$STAND_TSV"; return 1
  fi
  .venv/bin/python - "$label" "$json" "$STAND_TSV" <<'PY'
import json, sys
label, path, tsv = sys.argv[1], sys.argv[2], sys.argv[3]
d = json.load(open(path))
gait, trunk = d.get("gait", {}), d.get("trunk", {})
clr = gait.get("foot_clearance_body_z_m")
row = [
    label,
    f"{clr * 1000:.2f}" if isinstance(clr, (int, float)) else "nan",
    f"{trunk.get('tilt_mean_deg', float('nan')):.2f}",
    f"{trunk.get('z_mean', float('nan')) * 1000:.1f}",
    f"{d.get('net_yaw_deg', float('nan')):+.1f}",
    str(d.get("falls", d.get("n_resets", "-"))),
]
print("\t".join(row), file=open(tsv, "a"))
PY
}

printf 'label\tzero_cmd_clearance_mm\ttilt_deg\ttrunk_z_mm\tnet_yaw_deg\tfalls\n' > "$STAND_TSV"

for arm in "${ARMS[@]}"; do
  run="logs/rsl_rl/velocity/$arm"
  [ -d "$run" ] || { log "[$arm] no run dir; skipping"; continue; }
  mapfile -t cks < <(ls "$run"/model_*.pt 2>/dev/null | sort -t_ -k2 -n)
  [ ${#cks[@]} -gt 0 ] || { log "[$arm] no checkpoints"; continue; }
  last=$(printf '%s\n' "${cks[@]}" | tail -1)
  log "[$arm] ${#cks[@]} checkpoints; every ${STRIDE} iters + the last"
  for ck in "${cks[@]}"; do
    it=$(basename "$ck" .pt | sed 's/model_//')
    [ "$it" -eq 0 ] && continue                      # random policy is not a data point
    if [ "$((it % STRIDE))" -ne 0 ] && [ "$ck" != "$last" ]; then continue; fi
    label="${arm}_ck$(printf '%05d' "$it")"
    cut -f1 "$TSV" | grep -qx "$label" && continue   # idempotent
    onnx="$OUT/${label}.onnx"
    if [ ! -f "$onnx" ]; then
      .venv/bin/python scripts/export.py "$TASK" --checkpoint-file "$ck" --onnx-file "$onnx" \
        > "$OUT/${label}_export.log" 2>&1 || { log "[$label] export failed"; continue; }
    fi
    eval_one "$label" "$onnx" || log "[$label] eval failed"
    cut -f1 "$STAND_TSV" | grep -qx "$label" || measure_stand "$label" "$onnx" \
      || log "[$label] stand probe failed"
    log "[$label] $(tail -1 "$TSV" | cut -f5,6 | tr '\t' ' ')"
  done
done

# ── per-arm best + final (the KPI is "how straight does this RECIPE get") ────
# The standstill probe is joined on the SAME label, so one row answers all four of
# the user's original requirements: straight (|net yaw|), no marching in place
# (clearance at zero command), no marching at standstill (same, plus net yaw at zero
# command for the spin-in-place failure), head stability (from the gait JSONs).
log "=== best / final per arm (lower |net yaw| median = straighter) ==="
.venv/bin/python - "$TSV" "$STAND_TSV" "$OUT/summary.txt" <<'PY' | tee -a "$OUT/status.txt"
import csv, sys
def load(p):
    try: return list(csv.DictReader(open(p), delimiter="\t"))
    except Exception: return []
rows = [r for r in load(sys.argv[1]) if r.get("net_yaw_med_deg") not in ("-", "", None)]
stand = {r["label"]: r for r in load(sys.argv[2])} if len(sys.argv) > 2 else {}
def f(r, k):
    try: return float(str(r[k]).replace("+", ""))
    except Exception: return float("nan")
arms = {}
for r in rows: arms.setdefault(r["arm"], []).append(r)
hdr = (f"{'arm':26s} {'n':>2s} {'best_ck':>8s} {'|yaw|best':>9s} {'|yaw|final':>10s} "
       f"{'pathEff':>7s} {'speed':>7s} {'clear_mm':>8s} {'zeroYaw':>8s} {'falls':>5s}")
out = ["straightness @vx=0.2 (walker-conditioned not available here: all 64 envs)",
       hdr, "-" * len(hdr)]
for arm, rs in sorted(arms.items()):
    best = min(rs, key=lambda r: abs(f(r, "net_yaw_med_deg")))
    fin = max(rs, key=lambda r: int(r["ckpt"]))
    s = stand.get(best["label"], {})
    out.append(f"{arm:26s} {len(rs):2d} {('ck'+best['ckpt']):>8s} "
               f"{abs(f(best,'net_yaw_med_deg')):>9.1f} {abs(f(fin,'net_yaw_med_deg')):>10.1f} "
               f"{f(best,'path_eff_med'):>7.3f} {f(best,'speed_along_mps'):>7.4f} "
               f"{str(s.get('zero_cmd_clearance_mm','-')):>8s} {str(s.get('net_yaw_deg','-')):>8s} "
               f"{str(best.get('falls','-')):>5s}")
out += ["", "zero-command (standstill) probe, keyed by the BEST-straightness checkpoint:",
        "  clear_mm: nan/blank = no swing detected (NOT marching); >8 mm = marching",
        "  zeroYaw : net yaw over 20 s at zero command (large = spins in place)"]
text = "\n".join(out)
print(text)
open(sys.argv[3], "w").write(text + "\n")
PY

log "SWEEP COMPLETE; $TSV ; $STAND_TSV"
