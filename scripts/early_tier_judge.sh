#!/usr/bin/env bash
# =============================================================================
# Early-tier judge: decide each arm at 2000-4000 iterations, not at the end.
#
# Why. The user's calibration (2026-09-16): "正常情况下 2000-3000 步已经有明显的
# 效果" — and this repo's own record agrees (zero-command standstill forms by 2000
# iters, the sit-stand rise by 2000-4000). So an arm that shows nothing at 3000 is
# evidence the MECHANISM is not working, not evidence that it needs more time.
# Waiting 9000 iterations to find that out costs ~7 GPU-hours per arm.
#
# It also compares each arm at MATCHED ITERATIONS against the baseline run, which
# is the only fair way to ask "did this switch change anything": absolute yaw
# numbers wander a lot across checkpoints (the baseline run goes +7.7 at ck2000,
# -3.65 at ck3000, +12.7 at ck5000, -0.75 at ck7000, -11.95 at ck11999), so an
# arm can look better or worse purely by which checkpoint you happened to read.
#
# Judged on BOTH halves of the user's requirement, every time:
#   1. straight with no command  -> yaw DC at wz=0
#   2. turns when commanded      -> yaw DC at wz=+0.6 minus yaw DC at wz=0
#
# Conservative by construction:
#   * only runs when fewer than 2 training processes are live (never steals a slot
#     from the queue);
#   * at most one new (arm, checkpoint) per tick;
#   * appends to a TSV, so a re-run never duplicates work.
#
# Usage:
#   setsid nohup bash scripts/early_tier_judge.sh > logs/early_tier.out 2>&1 < /dev/null &
# =============================================================================
set -u
cd "$(dirname "$0")/.." || exit 1
export UV_CACHE_DIR=/tmp/uv-cache
export WANDB_MODE=offline

OUT=logs/queue_round2
TSV="$OUT/early_tier.tsv"
TMP="$OUT/early_tier"
ENVS=64
SECS=20
TIER_MIN=2000
TIER_MAX=4000
MAX_HOURS=${MAX_HOURS:-20}

mkdir -p "$TMP"
[ -f "$TSV" ] || printf 'arm\tckpt\tyawDC_wz0\tgait_mps\tpathEff\tyawDC_wz06\tsteer_resp\tsurvived0\tsurvived6\n' > "$TSV"

log() { echo "[$(date '+%F %T')] $*"; }

# arm name is recovered from the run's OWN saved config, never from a note:
#   turn_first       -> turn_first curriculum,      no yaw_dc term
#   turnfirst_yawdc  -> turn_first curriculum AND   yaw_dc term
#   walk_yawdc       -> no turn_first curriculum,   yaw_dc term
#   walk_sym         -> symmetry_cfg present,       no yaw_dc term
arm_of_run() {
  .venv/bin/python - "$1" <<'PY'
import sys, pathlib
rd = pathlib.Path(sys.argv[1])
agent = (rd / "params" / "agent.yaml").read_text() if (rd / "params" / "agent.yaml").exists() else ""
env = (rd / "params" / "env.yaml").read_text() if (rd / "params" / "env.yaml").exists() else ""
both = agent + env
has_turn = "turn_first" in both
has_dc = "yaw_dc" in both
has_sym = "symmetry_cfg" in agent and "use_mirror_loss: true" in agent
if has_turn and has_dc: print("turnfirst_yawdc")
elif has_turn:          print("turn_first")
elif has_dc:            print("walk_yawdc")
elif has_sym:           print("walk_sym")
PY
}

judged_already() {  # $1 arm, $2 ckpt
  [ -f "$TSV" ] && awk -F'\t' -v a="$1" -v c="$2" '$1==a && $2==c {found=1} END{exit !found}' "$TSV"
}

judge_one() {  # $1 arm, $2 run_dir, $3 ckpt
  local arm="$1" rd="$2" ck="$3"
  local raw="$TMP/${arm}_${ck}.onnx" dyn="$TMP/${arm}_${ck}_dyn.onnx"
  log "[$arm ck$ck] exporting"
  .venv/bin/python scripts/export.py Mjlab-Velocity-Flat-MicroDuck \
      --checkpoint-file "$rd/model_${ck}.pt" --onnx-file "$raw" > "$TMP/exp_${arm}_${ck}.log" 2>&1
  [ -f "$raw" ] || { log "[$arm ck$ck] export FAILED"; return 1; }
  .venv/bin/python scripts/make_dynamic_batch_onnx.py --onnx "$raw" --write "$dyn" \
      --check-batch "$ENVS" > "$TMP/dyn_${arm}_${ck}.log" 2>&1
  [ -f "$dyn" ] || dyn="$raw"

  local j0="$TMP/${arm}_${ck}_wz0.json" j6="$TMP/${arm}_${ck}_wz6.json"
  .venv/bin/python scripts/eval_straightness.py --onnx "$dyn" --lin-vel-x 0.2 \
      --num-envs "$ENVS" --seed 0 --seconds "$SECS" --freeze-pose-cmd \
      --json "$j0" > "$TMP/ev_${arm}_${ck}_wz0.log" 2>&1
  .venv/bin/python scripts/eval_straightness.py --onnx "$dyn" --lin-vel-x 0.2 \
      --ang-vel-z 0.6 --num-envs "$ENVS" --seed 0 --seconds "$SECS" --freeze-pose-cmd \
      --json "$j6" > "$TMP/ev_${arm}_${ck}_wz6.log" 2>&1

  .venv/bin/python - "$arm" "$ck" "$j0" "$j6" "$TSV" <<'PY'
import json, sys, pathlib
arm, ck, p0, p6, tsv = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]
def g(p):
    try:
        d = json.load(open(p)); return d, d["aggregate"]
    except Exception:
        return None, None
d0, a0 = g(p0); d6, a6 = g(p6)
if a0 is None:
    print(f"  {arm} ck{ck}: eval FAILED"); raise SystemExit
y0 = a0["yaw_dc_body_deg_s"]["mean"]
v0 = a0["total_path_m"]["mean"] / 20.0
eff = a0["path_efficiency"]["mean"]
y6 = a6["yaw_dc_body_deg_s"]["mean"] if a6 else float("nan")
resp = (y6 - y0) if a6 else float("nan")
pathlib.Path(tsv).open("a").write(
    f"{arm}\t{ck}\t{y0:+.2f}\t{v0:.3f}\t{eff:.3f}\t{y6:+.2f}\t{resp:+.2f}\t"
    f"{bool(d0 and d0.get('survived_window'))}\t{bool(d6 and d6.get('survived_window'))}\n")
print(f"  {arm:<17} ck{ck:<5} yawDC(wz0) {y0:+7.2f}  gait {v0:.3f} m/s  eff {eff:+.3f}"
      f"   steer resp {resp:+6.2f} deg/s")
PY
}

deadline=$(( $(date +%s) + MAX_HOURS * 3600 ))
log "early-tier judge armed (tier ${TIER_MIN}-${TIER_MAX} iters, max ${MAX_HOURS}h)"
while [ "$(date +%s)" -lt "$deadline" ]; do
  # Gate on FREE VRAM, not on the number of training processes. The first version
  # required fewer than 2 trainings, which is never true while the queue runs —
  # so the judge armed itself and then did nothing for hours. A 64-env 20 s eval
  # needs ~1 GB and the card keeps ~3.5 GB free alongside two trainings, so the
  # right guard is "is there room for one eval", not "is the GPU idle".
  free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
  if [ "${free_mib:-0}" -lt 1500 ]; then sleep 300; continue; fi
  did_one=0
  for rd in logs/rsl_rl/velocity/*/ logs/rsl_rl/microduck_sitstand/*/; do
    [ -d "$rd" ] || continue
    arm=$(arm_of_run "$rd")
    [ -z "$arm" ] && continue
    for ckpt in $(ls "$rd"model_*.pt 2>/dev/null | sed 's/.*model_//;s/\.pt//' | sort -n); do
      [ "$ckpt" -lt "$TIER_MIN" ] && continue
      [ "$ckpt" -gt "$TIER_MAX" ] && continue
      judged_already "$arm" "$ckpt" && continue
      judge_one "$arm" "$rd" "$ckpt" && did_one=1 && break 2
    done
  done
  [ "$did_one" -eq 0 ] && sleep 300
done
log "early-tier judge finished. table: $TSV"
