#!/usr/bin/env bash
# Real FLIGHT fraction for the new arms, using agent D's validated probe.
#
# Why not my own numbers: I had been deriving "flight = 1 - 2*duty" from
# gait_forensics' `duty_high_frac`, and that is WRONG -- reading the source
# (scripts/gait_forensics.py:433) shows `above = h > thresh` where h is the LEFT
# FOOT HEIGHT and thresh is the midpoint of its own 5/95 percentiles. It is a
# left-foot lift-time fraction that sits near 0.5 for ANY gait by construction,
# not a contact duty. Agent D measured contact properly with its own probe, and
# that probe is what this script reuses (no new hand-written probe: the repo's
# rule is that a probe must reproduce an existing measurement first, and this one
# already did -- D cross-checked it against eval_straightness).
#
# The two reference rows are the self-check: official must come out ~0.000 flight
# and ck7000 ~0.119, the numbers D published. If they do not, the harness is wrong
# and no row below means anything.
#
# Usage: nohup setsid bash scripts/measure_flight_all.sh > logs/flight_all.out 2>&1 &
set -u
cd "$(dirname "$0")/.." || exit 1
OUT=${OUT:-logs/flight}
mkdir -p "$OUT"
export UV_CACHE_DIR=/tmp/uv-cache

PROBE=docs/forensics_2026-09-18/agent_D_probe_p64.py
TASK="Mjlab-Velocity-Flat-MicroDuck"

run() {  # label onnx [extra probe args...]
  local label="$1" onnx="$2"; shift 2
  [ -f "$onnx" ] || { echo "[!] $label: missing $onnx"; return 1; }
  echo "[$(date +%H:%M:%S)] probing $label"
  .venv/bin/python "$PROBE" --onnx "$onnx" --tag "$label" --task "$TASK" \
    --lin-vel-x 0.2 --num-envs 64 --seed 0 --seconds 20 \
    --out "$OUT/$label.npz" "$@" > "$OUT/$label.probe.log" 2>&1 ||
    { echo "[!] $label probe failed (see $OUT/$label.probe.log)"; return 1; }
}

# references first: they calibrate the harness against D's published numbers
run official /home/joyandai/microduck/policies/alpha_walking.onnx \
  --override-model preswap --override-actuator xl330
run ck7000 walk_hd1910_7000.onnx
# the arms under test
for a in turn_official_s42 turn_official_s43 ours_flight_s42 ours_flight_s43 \
         ours_flight_s44 ours_head_s43 ours_head_s42; do
  run "$a" "logs/queue_overnight/${a}.onnx"
done

# analyse with D's own function (exec'd up to its hardcoded demo loop)
.venv/bin/python - "$OUT" <<'PY'
import sys, json, pathlib
src = pathlib.Path("docs/forensics_2026-09-18/agent_D_gait_correct.py").read_text()
src = src.split("for n,f in [")[0]          # keep the definitions, drop the demo
ns = {}
exec(compile(src, "agent_D_gait_correct.py", "exec"), ns)
analyse = ns["analyse"]
out = pathlib.Path(sys.argv[1])
rows = []
for npz in sorted(out.glob("*.npz")):
    try:
        rows.append(analyse(npz.stem, str(npz)))
    except Exception as e:
        print(f"[!] {npz.stem}: {type(e).__name__}: {e}")
print()
print(f"{'policy':22s}{'flight':>9s}{'double':>9s}{'duty_L':>8s}{'duty_R':>8s}"
      f"{'step Hz':>9s}{'clr_L mm':>10s}{'walkers':>9s}{'walk_duty':>11s}")
for r in rows:
    print(f"{r['name']:22s}{r['flight']:9.4f}{r['double_support']:9.3f}{r['duty_L']:8.3f}"
          f"{r['duty_R']:8.3f}{r['step_hz_per_foot']:9.2f}{r['clr_L']:10.1f}"
          f"{r.get('walkers_n', 0):9d}{r.get('walkers_duty', float('nan')):11.3f}")
json.dump(rows, open(out / "flight_summary.json", "w"), indent=1)
print(f"\n[self-check] official flight should be ~0.000 and ck7000 ~0.119 "
      f"(agent D's published numbers)")
PY
