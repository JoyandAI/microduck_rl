#!/usr/bin/env bash
# Sweep every checkpoint of ONE existing run for the metric that matters: the net
# yaw rate. The delivered walk_hd1910_12000.onnx can only be improved "for free"
# if some checkpoint of its own run happens to sit in a straighter basin — the
# official deliverable IS a hand-picked checkpoint (BEST_alpha_walking_rough.onnx),
# and this repo has already measured within-run checkpoint spread of
# 0.043..0.489 path efficiency. So: measure before training anything.
#
# Usage: bash scripts/sweep_run_yaw.sh <run_dir> [seconds] [num_envs]
set -u
cd "$(dirname "$0")/.." || exit 1
export UV_CACHE_DIR=/tmp/uv-cache WANDB_MODE=offline

RUN=${1:?usage: sweep_run_yaw.sh <run_dir> [seconds] [num_envs]}
SECS=${2:-20}
ENVS=${3:-64}
TMP=logs/ckpt_yaw_sweep
mkdir -p "$TMP"
OUT="$TMP/$(basename "$RUN").tsv"

[ -f "$OUT" ] || printf 'ckpt\tyawDC_deg_s\tpathEff\tturnR_m\tspeed_mps\twindow_s\tsurvived\n' > "$OUT"

for ck in $(ls "$RUN"/model_*.pt | sort -t_ -k2 -n); do
  it=$(basename "$ck" .pt | sed 's/model_//')
  onnx="$TMP/$(basename "$RUN")_$it.onnx"
  dyn="$TMP/$(basename "$RUN")_${it}_dyn.onnx"
  js="$TMP/$(basename "$RUN")_$it.json"
  if [ ! -f "$dyn" ]; then
    .venv/bin/python scripts/export.py Mjlab-Velocity-Flat-MicroDuck \
        --checkpoint-file "$ck" --onnx-file "$onnx" > "$TMP/exp_$it.log" 2>&1
    [ -f "$onnx" ] || { echo "[!] ck$it export failed"; continue; }
    .venv/bin/python scripts/make_dynamic_batch_onnx.py --onnx "$onnx" \
        --write "$dyn" --check-batch "$ENVS" > "$TMP/dyn_$it.log" 2>&1
    [ -f "$dyn" ] || { echo "[!] ck$it dyn failed"; continue; }
  fi
  [ -f "$js" ] || .venv/bin/python scripts/eval_straightness.py --onnx "$dyn" \
      --lin-vel-x 0.2 --num-envs "$ENVS" --seed 0 --seconds "$SECS" \
      --freeze-pose-cmd --json "$js" > "$TMP/eval_$it.log" 2>&1
  python3 - "$js" "$it" "$OUT" <<'PY'
import json, math, sys, pathlib
js, it, out = sys.argv[1], sys.argv[2], sys.argv[3]
try:
    d = json.load(open(js)); a = d["aggregate"]
except Exception:
    print(f"  ck{it}: no result"); raise SystemExit
w = a["yaw_dc_body_deg_s"]["mean"]
v = a["total_path_m"]["mean"] / max(d["seconds"], 1e-6)
R = v / abs(math.radians(w)) if abs(w) > 1e-9 else float("inf")
win = d.get("median_window_s", 0.0)
surv = d.get("survived_window", False)
pathlib.Path(out).open("a").write(
    f"{it}\t{w:+.3f}\t{a['path_efficiency']['mean']:.3f}\t{R:.2f}\t{v:.3f}\t{win:.1f}\t{surv}\n")
print(f"  ck{it:<7} yawDC {w:+7.2f} deg/s   R {R:7.2f} m   speed {v:.3f}   "
      f"eff {a['path_efficiency']['mean']:.3f}   win {win:.1f}s")
PY
done
echo
echo "[i] -> $OUT"
python3 - "$OUT" <<'PY'
import sys, csv
rows = list(csv.DictReader(open(sys.argv[1]), delimiter="\t"))
ok = [r for r in rows if r["survived"] == "True"]
ok.sort(key=lambda r: abs(float(r["yawDC_deg_s"])))
print("best checkpoints by |yaw DC| (only those that survived the window):")
for r in ok[:8]:
    print(f"  ck{r['ckpt']:<7} yawDC {float(r['yawDC_deg_s']):+7.2f}   "
          f"R {float(r['turnR_m']):7.2f} m   speed {float(r['speed_mps']):.3f}   "
          f"eff {float(r['pathEff']):.3f}")
if not ok:
    print("  none survived — every checkpoint of this run fails to walk the window")
PY
