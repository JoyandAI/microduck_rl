#!/usr/bin/env bash
# Per-half-step YAW IMPULSE for the new arms, using agent D's validated analysis.
#
# WHY THIS AND NOT FLIGHT: the flight penalty succeeded on its own metric (flight
# 0.229 -> 0.031, 3/3 seeds) but did NOT buy straightness across the arms we have
# (ours_flight_s43 has the 2nd-lowest flight AND the 2nd-worst net yaw; turn_official
# has 0.24 flight and the BEST net yaw of the new arms). So flight is not the causal
# quantity. Agent D identified the causal one: the net yaw impulse per half-step --
# official 0.46 deg vs ck7000 6.35 deg (14x) -- i.e. the official is straight because
# its gait does not PRODUCE a disturbance, while we produce a large one and cancel it
# down to a few percent, leaving a large random residual.
#
# This script regenerates per-frame CSVs for the new arms and runs D's own analysis
# on them plus the stored reference CSVs (logs/axis/csv_*.csv, which are P64 dumps).
#
# Usage: nohup setsid bash scripts/measure_impulse_all.sh > logs/impulse_all.out 2>&1 &
set -u
cd "$(dirname "$0")/.." || exit 1
OUT=${OUT:-logs/impulse}
mkdir -p "$OUT"
export UV_CACHE_DIR=/tmp/uv-cache
TASK="Mjlab-Velocity-Flat-MicroDuck"

gen() {  # label onnx
  local label="$1" onnx="$2"
  [ -f "$OUT/$label.csv" ] && return 0
  [ -f "$onnx" ] || { echo "[!] $label: no $onnx"; return 1; }
  echo "[$(date +%H:%M:%S)] generating CSV for $label"
  .venv/bin/python scripts/eval_straightness.py --onnx "$onnx" --task "$TASK" \
    --lin-vel-x 0.2 --num-envs 64 --seed 0 --seconds 20 --freeze-pose-cmd \
    --csv "$OUT/$label.csv" --json "$OUT/$label.json" \
    > "$OUT/$label.log" 2>&1 || { echo "[!] $label eval failed"; return 1; }
}

# references: reuse the stored P64 dumps so the harness is self-checked against
# D's published numbers (official 0.46, ck7000 6.35 deg per half-step)
for a in official ck7000; do
  [ -f "logs/axis/csv_${a}_0.2.csv" ] && cp -n "logs/axis/csv_${a}_0.2.csv" "$OUT/$a.csv"
done
gen align_fwd logs/queue_align/align_fwd.onnx
for a in ours_head_s43 ours_flight_s42 ours_flight_s43 ours_flight_s44 \
         turn_ours_s42 turn_official_s42; do
  gen "$a" "logs/queue_overnight/${a}.onnx"
done

.venv/bin/python - "$OUT" <<'PY'
import sys, pathlib, json
src = pathlib.Path("docs/forensics_2026-09-18/agent_D_phase_impulse.py").read_text()
src = src.split('if __name__ == "__main__":')[0]
ns = {}
exec(compile(src, "agent_D_phase_impulse.py", "exec"), ns)
analyse, rep = ns["analyse"], ns["rep"]
out = pathlib.Path(sys.argv[1])
print()
for csv in sorted(out.glob("*.csv")):
    try:
        mean, med = analyse(str(csv))
    except Exception as e:
        print(f"{csv.stem:22s} ERROR {type(e).__name__}: {e}"); continue
    keys = ("imp_L","imp_R","imp_net","ac_lock","yaw_dc","speed_along","duty_L","duty_R")
    vals = " ".join(f"{k}={mean.get(k, float('nan')):+.3f}" for k in keys if k in mean)
    print(f"{csv.stem:22s} {vals}")
print("\n[self-check] official imp_L ~0.46 deg and ck7000 ~6.35 deg per half-step "
      "(agent D's published numbers)")
PY
