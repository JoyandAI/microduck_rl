#!/usr/bin/env bash
# Does the official policy WALK STRAIGHT and STEER by rate control — or does it
# carry a natural curvature and "steer" by re-using it?
#
# User's hypothesis (2026-09-20): "有沒有可能官方也不是完全走直线，利用合理的转弯当转向效果"
# Two predictions separate the hypotheses on the SAME 64-env frozen protocol:
#   RATE CONTROL      yaw_rate(wz) is linear and passes through the origin, and the
#                     gain is symmetric about wz = 0.
#   CURVATURE RE-USE   yaw_rate(wz) has a nonzero INTERCEPT (a natural curvature the
#                     gait already has) and/or an asymmetric gain, and the stance-foot
#                     geometry changes with wz (lateral placement / torsion).
# The evaluator already records the gait side of this per env: `foot_pair_torsion_deg`,
# `hip_yaw_{L,R}_rel_deg`, `lateral_per_metre`, `turning_radius_m`, duty and clearance.
#
# Physics: the repo's default (HD-1910 m5) — the same machine every one of our arms was
# measured on, so the numbers are directly comparable. The earlier official figure
# (med|net yaw| 2.8 deg) used the xl330 preset instead, i.e. a different robot.
#
# Usage: bash scripts/probe_official_steering.sh [--launch]
set -u
cd "$(dirname "$0")/.." || exit 1
OUT=logs/steer_curve
mkdir -p "$OUT"
TASK="Mjlab-Velocity-Flat-MicroDuck"
OFFICIAL=/home/joyandai/microduck/policies/alpha_walking.onnx
export UV_CACHE_DIR=/tmp/uv-cache MPLCONFIGDIR=/tmp/matplotlib WANDB_MODE=offline
log() { printf '[%(%F %T)T] %s\n' -1 "$*" | tee -a "$OUT/status.txt" >&2; }

run() {  # label onnx vx wz
  local label="$1" onnx="$2" vx="$3" wz="$4"
  local json="$OUT/${label}_vx${vx}_wz${wz}.json"
  [ -f "$json" ] && { log "[$label vx=$vx wz=$wz] cached"; return 0; }
  .venv/bin/python scripts/eval_straightness.py --onnx "$onnx" --task "$TASK" \
    --lin-vel-x "$vx" --ang-vel-z "$wz" --num-envs 64 --seed 0 --seconds 20 \
    --freeze-pose-cmd --json "$json" > "$OUT/${label}_vx${vx}_wz${wz}.log" 2>&1 \
    || { log "[$label vx=$vx wz=$wz] EVAL FAILED"; return 1; }
  log "[$label vx=$vx wz=$wz] done"
}

# ── the sweep ───────────────────────────────────────────────────────────────
# 1. official, walking, yaw command both signs + the interior points (linearity /
#    intercept / symmetry). 0.1 rad/s is included because a curvature-re-use gait
#    would show a DEAD ZONE or a sign error there, which a pure rate channel would not.
for wz in -1.0 -0.6 -0.3 -0.1 0.1 0.3 0.6 1.0; do
  run official "$OFFICIAL" 0.2 "$wz"
done
# 2. official, standing, yaw command: can it turn on the spot at all? (deployment)
for wz in -0.6 0.6; do
  run official_stand "$OFFICIAL" 0.0 "$wz"
done
# 3. our two reference arms on the SAME sweep, for contrast: the best straight walker
#    we ever shipped, and the best of the new flight arm.
for wz in -0.6 -0.1 0.1 0.6; do
  run ck7000 walk_hd1910_7000.onnx 0.2 "$wz"
  run ours_flight_s42 logs/queue_routeA/ours_flight_s42.onnx 0.2 "$wz"
done

# ── summary: the two hypotheses' discriminators ─────────────────────────────
.venv/bin/python - "$OUT" <<'PY' | tee "$OUT/SUMMARY.txt"
import glob, json, os, re, statistics as st, sys
out = sys.argv[1]
rows = {}
for f in sorted(glob.glob(os.path.join(out, "*.json"))):
    m = re.match(r"(.+)_vx([-\d.]+)_wz([-\d.]+)\.json$", os.path.basename(f))
    if not m:
        continue
    label, vx, wz = m.group(1), float(m.group(2)), float(m.group(3))
    d = json.load(open(f))
    per = d["per_env"]
    yaw = [e["yaw_dc_body_deg_s"] for e in per if e.get("yaw_dc_body_deg_s") is not None]
    rows[(label, vx, wz)] = {
        "rate": st.mean(yaw),
        "rate_bias": (d["aggregate"].get("yaw_rate_bias_deg_s") or {}).get("mean"),
        "net": st.median([abs(e["net_yaw_deg"]) for e in per]),
        "torsion": st.median([abs(e.get("foot_pair_torsion_deg", 0.0)) for e in per]),
        "hipyaw": st.median([abs(e.get("hip_yaw_L_rel_deg", 0.0)) + abs(e.get("hip_yaw_R_rel_deg", 0.0)) for e in per]),
        "lat_per_m": st.median([abs(e.get("lateral_per_metre", 0.0)) for e in per]),
        "speed": st.median([e["speed_along_mps"] for e in per]),
        "falls": d.get("falls"),
        "window": d.get("median_window_s"),
        "full": d.get("frac_envs_full_window"),
    }
CMD = 34.377  # 0.6 rad/s
print("cmd yaw rate = 0.6 rad/s = 34.377 deg/s;  rate = mean of per-env yaw_dc_body_deg_s")
for label in sorted({k[0] for k in rows}):
    print(f"\n=== {label} ===")
    print(f"{'vx':>5s} {'wz':>6s} {'yaw rate':>9s} {'net|yaw|':>9s} {'gain%':>7s} "
          f"{'torsion':>8s} {'hipyaw':>7s} {'lat/m':>7s} {'speed':>7s} {'falls':>5s} {'win':>5s}")
    for (lab, vx, wz) in sorted(k for k in rows if k[0] == label):
        r = rows[(lab, vx, wz)]
        gain = 100 * r["rate"] / (wz * 57.2958) if wz else float("nan")
        print(f"{vx:>5.1f} {wz:>6.2f} {r['rate']:>9.2f} {r['net']:>9.1f} {gain:>6.1f}% "
              f"{r['torsion']:>8.2f} {r['hipyaw']:>7.2f} {r['lat_per_m']:>7.3f} "
              f"{r['speed']:>7.4f} {str(r['falls']):>5s} {str(r['window']):>5s}")
    # intercept test: fit rate = k*wz + b over the walking sweep
    pts = [(wz, r["rate"]) for (lab, vx, wz), r in rows.items() if lab == label and vx == 0.2 and wz != 0]
    if len(pts) >= 4:
        n = len(pts); sx = sum(p[0] for p in pts); sy = sum(p[1] for p in pts)
        sxx = sum(p[0] ** 2 for p in pts); sxy = sum(p[0] * p[1] for p in pts)
        k = (n * sxy - sx * sy) / (n * sxx - sx * sx); b = (sy - k * sx) / n
        pos = [r["rate"] for (lab, vx, wz), r in rows.items() if lab == label and wz > 0]
        neg = [r["rate"] for (lab, vx, wz), r in rows.items() if lab == label and wz < 0]
        print(f"  fit: yaw_rate = {k:.2f}*wz + {b:+.2f} deg/s   (intercept {b:+.2f} deg/s; "
              f"natural turning radius {abs(0.2 / (b / 57.2958)) if abs(b) > 1e-9 else float('inf'):.1f} m)")
        print(f"  symmetry: mean(+)={st.mean(pos):+.2f}  mean(-)={st.mean(neg):+.2f}  "
              f"ratio |+|/|-| = {abs(st.mean(pos)) / abs(st.mean(neg)) if st.mean(neg) else float('nan'):.2f}")
PY
log "steering curve done: $OUT/SUMMARY.txt"
