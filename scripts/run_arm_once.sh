#!/usr/bin/env bash
# Run ONE arm at ONE seed: smoke -> verify switches -> train -> export -> P64 evals.
# Reusable by any driver queue so the recipe/verification logic lives in one place.
#
# Usage: bash scripts/run_arm_once.sh <label> <seed> "<ENV VARS>" "<terms to verify>"
# Prints the run dir on stdout (last line); everything else goes to the log.
set -u

cd "$(dirname "$0")/.." || exit 1
LABEL="$1"; SEED="$2"; ARM_ENV="$3"; WANT="$4"

ITERS=${ITERS:-9000}
ENVS=${ENVS:-2048}
SMOKE_ITERS=${SMOKE_ITERS:-5}
OUT=${OUT:-logs/queue_routeA}
TASK="Mjlab-Velocity-Flat-MicroDuck"
RUNNAME="${LABEL}_s${SEED}"
mkdir -p "$OUT"

export UV_CACHE_DIR=/tmp/uv-cache MPLCONFIGDIR=/tmp/matplotlib WANDB_MODE=offline
export MICRODUCK_TASK_NAMED_RUNS=1   # opt-in; without it runs collide on a timestamp

log() { printf '[%(%F %T)T] %s\n' -1 "[$RUNNAME] $*" | tee -a "$OUT/status.txt" >&2; }
fingerprint() {
  .venv/bin/python - <<'PY'
from mjlab_microduck.train_hook import actuator_params_fingerprint
print(actuator_params_fingerprint()[0])
PY
}

# ── GPU-driver gate + smoke retry ───────────────────────────────────────────
# The driver died twice on 2026-09-17/18 ("Failed to initialize NVML", CUDA Error
# 304). A running training survives (it holds its context) but a NEW process cannot
# initialise, so a queue that treats the first smoke failure as fatal would throw
# away the rest of the night. Wait for the driver, then retry once.
wait_driver() {
  local waited=0
  while ! nvidia-smi --query-gpu=memory.free --format=csv,noheader >/dev/null 2>&1; do
    [ $((waited % 300)) -eq 0 ] && log "waiting for the GPU driver (${waited}s)"
    sleep 30; waited=$((waited + 30))
    [ "$waited" -ge 7200 ] && { log "driver unavailable for 2 h"; return 1; }
  done
  return 0
}

# ── smoke + switch verification ─────────────────────────────────────────────
SLOG="$OUT/${RUNNAME}_smoke.log"
log "smoke: 64 envs x ${SMOKE_ITERS}"
env $ARM_ENV MICRODUCK_RUN_DIR_NAME="${RUNNAME}_smoke" \
  .venv/bin/python -m mjlab_microduck.train_cli "$TASK" \
    --env.scene.num-envs 64 --agent.max_iterations "$SMOKE_ITERS" \
    --agent.seed "$SEED" > "$SLOG" 2>&1
rc=$?
if [ "$rc" -ne 0 ] || grep -q 'Traceback (most recent call' "$SLOG"; then
  if grep -qE 'NVML|cudaGetDeviceCount|Error 304|CUDA initialization' "$SLOG"; then
    log "smoke hit the flaky GPU driver; waiting and retrying once"
    wait_driver || exit 1
    env $ARM_ENV MICRODUCK_RUN_DIR_NAME="${RUNNAME}_smoke" \
      .venv/bin/python -m mjlab_microduck.train_cli "$TASK" \
        --env.scene.num-envs 64 --agent.max_iterations "$SMOKE_ITERS" \
        --agent.seed "$SEED" > "$SLOG" 2>&1
    rc=$?
  fi
fi
{ [ "$rc" -ne 0 ] || grep -q 'Traceback (most recent call' "$SLOG"; } && {
  log "SMOKE FAILED rc=${rc}"; exit 1; }
missing=""
for t in $WANT; do grep -q "$t" "$SLOG" || missing="$missing $t"; done
[ -n "$missing" ] && { log "SMOKE missing:$missing"; exit 1; }
grep -q "run directory named by task: ${RUNNAME}_smoke" "$SLOG" || {
  log "FATAL: task-named run dirs did not engage"; exit 1; }
log "smoke passed; verified:$WANT"

# ── train ───────────────────────────────────────────────────────────────────
FP_BEFORE=$(fingerprint)
TLOG="$OUT/${RUNNAME}_train.log"
T0=$(date +%s)   # arm start; the guard below needs it because TLOG's mtime is its LAST write
log "TRAIN start: ${ITERS} iters @ ${ENVS} seed=${SEED}; fingerprint=${FP_BEFORE}"
env $ARM_ENV MICRODUCK_RUN_DIR_NAME="$RUNNAME" \
  .venv/bin/python -m mjlab_microduck.train_cli "$TASK" \
    --env.scene.num-envs "$ENVS" --agent.max_iterations "$ITERS" \
    --agent.seed "$SEED" --agent.save_interval 500 > "$TLOG" 2>&1
rc=$?
FP_AFTER=$(fingerprint)
# The TRAINING LOG is authoritative, not the conventional path (2026-09-20).
# `MICRODUCK_TASK_NAMED_RUNS=1` silently renames a colliding run to `<name>_r2`
# (that is how the overnight re-runs of ours_flight_s42/43/44 became
# ours_flight_s*_r2). The old `[ -d "$run_dir" ] || parse-the-log` order kept the
# STALE path whenever the label had been trained before, so `ck=$(ls -t ...)`
# exported the PREVIOUS run's last checkpoint: queue_overnight/ours_flight_s4*.onnx
# came out byte-identical to queue_routeA's, the three _r2 runs (~9 GPU-h) were
# never evaluated, and their training metrics (from $TLOG, which WAS the new run)
# described a different policy than their own evals. Parse the log FIRST.
run_dir=$(grep -o 'Logging experiment in directory: .*' "$TLOG" | tail -1 | sed 's/.*directory: //')
[ -n "$run_dir" ] && [ -d "$run_dir" ] || run_dir="logs/rsl_rl/velocity/$RUNNAME"
# Guard: the checkpoint we are about to export must have been written by THIS arm.
# Compare against T0 (arm start), not against TLOG's mtime — the log's final flush
# happens ~30 ms AFTER the last checkpoint, so `-newer "$TLOG"` always fails.
ck_probe=$(ls -t "$run_dir"/model_*.pt 2>/dev/null | head -1)
if [ -z "$ck_probe" ] || [ "$(stat -c %Y "$ck_probe")" -lt "$T0" ]; then
  log "FATAL: newest checkpoint in $run_dir predates this arm — wrong run dir (stale collision)"; exit 1
fi
printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
  "$RUNNAME" "$TASK" "$ITERS" "$ENVS" "$rc" "${run_dir:--}" "$FP_BEFORE" "$FP_AFTER" >> "$OUT/results.tsv"
[ "$FP_BEFORE" != "$FP_AFTER" ] && { log "CONTAMINATED fingerprint"; exit 1; }
{ [ "$rc" -ne 0 ] || [ ! -d "$run_dir" ]; } && { log "TRAIN FAILED rc=${rc}"; exit 1; }
log "training complete: ${run_dir}"

# ── the quantities the reports say to watch (read from the run's own log) ────
log "watch: Episode_Reward/flight (route-A lever) + Metrics/yaw_dc_abs (wander)"
for pat in 'Episode_Reward/flight' 'Metrics/yaw_dc_abs' 'Metrics/air_time_mean'; do
  printf '    %s: %s\n' "$pat" "$(grep -o "${pat}: [-0-9.]*" "$TLOG" | tail -1 | sed 's/.*: //')" | tee -a "$OUT/status.txt"
done

# ── export + P64 evals (same protocol for every arm) ─────────────────────────
ck=$(ls -t "$run_dir"/model_*.pt 2>/dev/null | head -1)
onnx="$OUT/${RUNNAME}.onnx"
.venv/bin/python scripts/export.py "$TASK" --checkpoint-file "$ck" \
  --onnx-file "$onnx" > "$OUT/${RUNNAME}_export.log" 2>&1 || { log "export failed"; exit 1; }
log "onnx=${onnx}"
for vx in 0.2 0.0; do
  .venv/bin/python scripts/eval_straightness.py --onnx "$onnx" --task "$TASK" \
    --lin-vel-x "$vx" --num-envs 64 --seed 0 --seconds 20 --freeze-pose-cmd \
    --json "$OUT/${RUNNAME}_vx${vx}.json" > "$OUT/${RUNNAME}_vx${vx}.log" 2>&1 ||
    log "eval vx=$vx failed"
done
# ── standstill probe（2026-09-20）─────────────────────────────────────────────
# The `--lin-vel-x 0.0` eval above is NOT a standstill measurement: at zero command
# the evaluator drops an env whose recording is shorter than 20 samples (0.4 s), so
# `n_envs_insufficient_data = 64` means every env RESET inside 0.4 s — a fall, not
# "no data", and the whole number set comes back empty. The repo's validated
# standstill probe is gait_forensics at zero command (foot clearance: standing
# 0.5-4.2 mm, marching 19.4 mm; plus tilt/net-yaw for the "spins in place" and
# "falls over" failures). Measured 2026-09-20 on the head/stance family: 55-64
# resets per 20 s and 72 deg tilt, vs 1 reset for the flight arm and the references.
STAND_JSON="$OUT/${RUNNAME}_stand.json"
.venv/bin/python scripts/gait_forensics.py --onnx "$onnx" --task "$TASK" \
  --lin-vel-x 0.0 --warmup 2 --seconds 20 --head-cmd zero \
  --json-out "$STAND_JSON" > "$OUT/${RUNNAME}_stand.log" 2>&1 || log "stand probe failed"
if [ -f "$STAND_JSON" ]; then
  .venv/bin/python - "$STAND_JSON" "$RUNNAME" <<'PY2' | tee -a "$OUT/status.txt"
import json, sys
d = json.load(open(sys.argv[1])); t = d.get("trunk", {}); g = d.get("gait", {})
clr = g.get("foot_clearance_body_z_m")
print(f"    standstill: resets={d.get('resets')} tilt_max={t.get('tilt_max_deg', float('nan')):.1f} deg "
      f"clearance={(clr * 1000 if isinstance(clr, (int, float)) else float('nan')):.2f} mm")
PY2
fi
# ── steering probe (agent F's acceptance quantity for the turn arms) ─────────
# F measured the steering gain as ODD/command, ODD = (v(+wz) - v(-wz))/2, with a
# PAIRED three-command sweep on one checkpoint. Same protocol here, so the turn
# arms can be graded against F's numbers: our best = +11.3 %, official = +57.6 %,
# and F's pass bar for turn_first is >= 35 %.
if [ "${STEER_PROBE:-0}" = "1" ]; then
  log "steering probe: wz = -0.6 / 0.0 / +0.6 rad/s (P64, paired)"
  for w in -0.6 0.0 0.6; do
    .venv/bin/python scripts/eval_straightness.py --onnx "$onnx" --task "$TASK" \
      --lin-vel-x 0.2 --ang-vel-z "$w" --num-envs 64 --seed 0 --seconds 20 \
      --freeze-pose-cmd --json "$OUT/${RUNNAME}_wz${w}.json" \
      > "$OUT/${RUNNAME}_wz${w}.log" 2>&1 || log "steer probe wz=$w failed"
  done
  .venv/bin/python - "$OUT" "$RUNNAME" <<'PY2'
import json, sys, os
out, name = sys.argv[1], sys.argv[2]
def dc(tag):
    try:
        a = json.load(open(f"{out}/{name}_wz{tag}.json"))["aggregate"]
        v = a.get("yaw_dc_body_deg_s")
        return (v.get("mean") if isinstance(v, dict) else v)
    except Exception:
        return None
neg, zer, pos = dc("-0.6"), dc("0.0"), dc("0.6")
if None in (neg, zer, pos):
    print("    steering: incomplete data", file=sys.stderr); raise SystemExit
odd = (pos - neg) / 2.0
print(f"    ODD = {odd:+.2f} deg/s  (wz=+34.4 deg/s command => gain {odd/34.38*100:.1f} %)")
print(f"    [{'PASS' if odd/34.38 >= 0.35 else 'fail'}] F's bar for turn_first >= 35 %")
print(f"    [{'ok' if abs(zer) < 5 else 'CHECK'}] zero-command self-rotation {zer:+.2f} deg/s (bar <5)")
PY2
fi

# gait character at the walking command (contact duty => flight fraction)
.venv/bin/python scripts/gait_forensics.py --onnx "$onnx" --task "$TASK" \
  --lin-vel-x 0.2 --warmup 2 --seconds 12 --head-cmd zero \
  --json-out "$OUT/${RUNNAME}_gait.json" > "$OUT/${RUNNAME}_gait.log" 2>&1 ||
  log "gait forensics failed"
.venv/bin/python scripts/check_posture_acceptance.py --onnx "$onnx" --keep-json \
  > "$OUT/${RUNNAME}_acceptance.log" 2>&1 || log "acceptance has FAIL rows (expected)"
log "done"
printf '%s\n' "$run_dir"
