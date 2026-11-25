#!/usr/bin/env bash
# "Is it the SPEED that makes us run?" — deterministic, no training.
#
# User hypothesis (2026-09-18): "我们可以放慢速度，是不是因为速度太快导致的".
# It is directly testable on a FROZEN checkpoint: hold the policy and sweep only the
# commanded forward speed. If the flight phase and the per-step yaw impulse appear
# only above some speed, then the command range is the cause and the fix is a
# command/curriculum change (accepting a slower robot). If they are already there at
# 0.05 m/s, speed is NOT the cause and the gait must be shaped directly.
#
# Two validated tools, no hand-written probe:
#   eval_straightness.py  (64 envs, the quotable protocol) -> yaw DC, median net yaw,
#                          path efficiency, achieved speed, AC/DC (per-step impulse)
#   gait_forensics.py     (1 env, trend only)              -> contact duty + clearance
# Contact duty is the flight proxy: total contact coverage = duty_L + duty_R; a
# flightless walk has both feet sharing the cycle (duty ~0.44-0.59 each), while a run
# drops duty and leaves gaps. AGENTS.md: never quote 1-env magnitudes, only trends.
#
# Usage: nohup setsid bash scripts/probe_speed_vs_flight.sh > logs/probe_speed.out 2>&1 &
set -u
cd "$(dirname "$0")/.." || exit 1
OUT=logs/probe_speed
mkdir -p "$OUT"
export UV_CACHE_DIR=/tmp/uv-cache MPLCONFIGDIR=/tmp/matplotlib

TASK="Mjlab-Velocity-Flat-MicroDuck"
ENVS=${ENVS:-64}
SECONDS_WIN=${SECONDS_WIN:-20}
SPEEDS=${SPEEDS:-"0.05 0.10 0.15 0.20 0.30"}
POLICIES=${POLICIES:-"ck7000:walk_hd1910_7000.onnx"}

{
  printf 'policy\tcmd_vx\tmade_good_mps\tarc_mps\tyaw_dc\tyaw_ac\tduty_high\tclearance_mm\tnet_yaw_med\tpath_eff\n'
} > "$OUT/speed_vs_flight.tsv"

for pol in $POLICIES; do
  name="${pol%%:*}"; onnx="${pol#*:}"
  [ -f "$onnx" ] || { echo "[!] missing $onnx"; continue; }
  for vx in $SPEEDS; do
    j="$OUT/${name}_vx${vx}.json"
    .venv/bin/python scripts/eval_straightness.py --onnx "$onnx" --task "$TASK" \
      --lin-vel-x "$vx" --num-envs "$ENVS" --seed 0 --seconds "$SECONDS_WIN" \
      --freeze-pose-cmd --json "$j" > "$OUT/${name}_vx${vx}_walk.log" 2>&1
    g="$OUT/${name}_vx${vx}_forensics.json"
    .venv/bin/python scripts/gait_forensics.py --onnx "$onnx" --task "$TASK" \
      --lin-vel-x "$vx" --warmup 2 --seconds 12 --head-cmd zero \
      --json-out "$g" > "$OUT/${name}_vx${vx}_gait.log" 2>&1
    .venv/bin/python - "$name" "$vx" "$j" "$g" "$OUT/speed_vs_flight.tsv" <<'PY'
import json, sys
name, vx, jp, gp, tsv = sys.argv[1:6]
try:
    d = json.load(open(jp)); a = d.get("aggregate", {})
except Exception:
    print(f"{name}\t{vx}\t(eval missing)", file=open(tsv, "a")); raise SystemExit
try:
    g = json.load(open(gp)).get("gait", {})
except Exception:
    g = {}
def m(k):
    v = a.get(k); return v.get("mean") if isinstance(v, dict) else v
arc = None
try:
    arc = m("total_path_m") / 20.0 if m("total_path_m") else None
except Exception:
    pass
row = [name, vx,
       f"{m('speed_along_mps'):.4f}", f"{arc:.4f}" if arc else "-",
       f"{m('yaw_dc_body_deg_s'):+.3f}", f"{m('yaw_ac_lock_deg_s'):+.3f}",
       f"{g.get('duty_high_frac', float('nan')):.3f}",
       f"{(g.get('foot_clearance_body_z_m') or float('nan'))*1000:.1f}",
       f"{a.get('net_yaw_deg__median', float('nan')):+.1f}",
       f"{a.get('path_efficiency__median', float('nan')):.3f}"]
print("\t".join(row), file=open(tsv, "a"))
print("  " + "\t".join(row))
PY
  done
done

echo
echo "=== 速度 vs 步态（duty 越低越像跑；AC 是每步偏航冲量）==="
column -t -s$'\t' "$OUT/speed_vs_flight.tsv"
echo
echo "读法：若 duty 随速度单调下降、AC 单调上升 ⇒ **速度是因**；"
echo "      若在 0.05 m/s 就已经 duty 低/AC 大 ⇒ 速度不是因，需要直接塑造步态。"
