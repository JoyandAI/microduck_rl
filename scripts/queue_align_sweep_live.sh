#!/usr/bin/env bash
# LIVE straightness sweep — evaluate checkpoints AS THEY APPEAR while the
# alignment arms are still training (user request 2026-09-17: "扫").
#
# Costs: one 64-env eval (~1 min) + one marching measurement (~1 min) + one ONNX
# export (~20 s) per checkpoint per arm. At ~2.9 s/iter a 1000-iteration gap is
# ~48 min of training, so this adds roughly 5 % wall-clock contention and buys an
# early read on the KPI instead of waiting 7.5 h for model_9000.
#
# Shares the label scheme and the TSVs with scripts/queue_align_sweep.sh, which
# runs after training and SKIPS labels already present here — the two scripts
# together produce one table with no duplicated work.
#
# Usage: nohup setsid bash scripts/queue_align_sweep_live.sh > logs/queue_align_sweep_live.out 2>&1 &
set -u

cd "$(dirname "$0")/.." || exit 1
OUT=logs/queue_align/sweep
mkdir -p "$OUT"

export UV_CACHE_DIR=/tmp/uv-cache
export MPLCONFIGDIR=/tmp/matplotlib

TASK="Mjlab-Velocity-Flat-MicroDuck"
STRIDE=${STRIDE:-1000}
POLL_S=${POLL_S:-120}
ARMS=${ARMS:-"ours_head turn_ours turn_official ours_flight ours_stance ours_ztrack ours_flight_stance ours_yawdc"}

log() { printf '[%(%F %T)T] %s\n' -1 "$*" | tee -a "$OUT/live_status.txt" >&2; }

TSV="$OUT/straightness.tsv"
MARCH_TSV="$OUT/marching.tsv"
[ -f "$TSV" ] || printf 'label\tyaw_dc_deg_s\tnet_yaw_med_deg\tpath_eff_med\tmedian_window_s\tfull_window_frac\tfalls\tspeed_along_mps\ttilt_deg\n' > "$TSV"
[ -f "$MARCH_TSV" ] || printf 'label\tzero_cmd_clearance_mm\ttilt_deg\ttrunk_z_mm\tnet_yaw_deg\tfalls\n' > "$MARCH_TSV"

done_label() { cut -f1 "$TSV" | grep -qx "$1"; }

eval_one() {
  local label="$1" onnx="$2" json="$OUT/${label}_walk.json"
  .venv/bin/python scripts/eval_straightness.py --onnx "$onnx" --task "$TASK" \
    --lin-vel-x 0.2 --num-envs 64 --seed 0 --seconds 20 --freeze-pose-cmd \
    --json "$json" > "$OUT/${label}_walk.log" 2>&1 || { log "[$label] eval failed"; return 1; }
  .venv/bin/python - "$label" "$json" "$TSV" <<'PY'
import json, sys
label, path, tsv = sys.argv[1], sys.argv[2], sys.argv[3]
d = json.load(open(path)); agg = d.get("aggregate", {})
def g(k, sub="mean"):
    v = agg.get(k); return v.get(sub) if isinstance(v, dict) else v
print("\t".join([label, f"{g('yaw_rate_bias_deg_s'):+.3f}",
                 f"{agg.get('net_yaw_deg__median', float('nan')):+.1f}",
                 f"{agg.get('path_efficiency__median', float('nan')):.3f}",
                 f"{d.get('median_window_s', float('nan')):.2f}",
                 f"{d.get('frac_envs_full_window', float('nan')):.2f}",
                 str(d.get("falls")), f"{g('speed_along_mps'):.3f}",
                 f"{g('tilt_mean_deg'):.2f}"]), file=open(tsv, "a"))
PY
}

measure_marching() {
  local label="$1" onnx="$2" json="$OUT/${label}_stand_forensics.json"
  .venv/bin/python scripts/gait_forensics.py --onnx "$onnx" --task "$TASK" \
    --lin-vel-x 0.0 --warmup 2 --seconds 20 --head-cmd zero \
    --json-out "$json" > "$OUT/${label}_stand.log" 2>&1 || { printf '%s\t-\t-\t-\t-\t-\n' "$label" >> "$MARCH_TSV"; return 1; }
  .venv/bin/python - "$label" "$json" "$MARCH_TSV" <<'PY'
import json, sys
label, path, tsv = sys.argv[1], sys.argv[2], sys.argv[3]
d = json.load(open(path)); gait = d.get("gait", {}); trunk = d.get("trunk", {})
# key is foot_clearance_body_z_m (METRES) -- asking for `foot_clearance_mm` yields
# nan for every row, which is how the first version of this table came out empty.
print("\t".join([label, f"{(gait.get('foot_clearance_body_z_m') or float('nan')) * 1000:.2f}",
                 f"{trunk.get('tilt_mean_deg', float('nan')):.2f}",
                 f"{trunk.get('z_mean', float('nan')) * 1000:.1f}",
                 f"{d.get('net_yaw_deg', float('nan')):+.1f}",
                 str(d.get("falls", d.get("n_resets", "-")))]), file=open(tsv, "a"))
PY
}

report_so_far() {
  [ "$(wc -l < "$TSV")" -le 1 ] && return 0
  log "=== 走直排名（|偏航 DC| 升序；官方参照 ≈0.2 deg/s、净偏航中位 2.8°、效率中位 0.43）==="
  { printf 'label\tyaw_dc\tnet_yaw_med\tpath_eff\twindow\tsurv\tfalls\tspeed\ttilt\n'
    tail -n +2 "$TSV" | awk -F'\t' '{v=$2; if (v<0) v=-v; print v"\t"$0}' |
      sort -g | cut -f2-; } | column -t -s$'\t' | tee -a "$OUT/live_status.txt"
  [ "$(wc -l < "$MARCH_TSV")" -gt 1 ] && {
    log "=== 零命令原地踏步（判据 <8 mm；官方 0.00、旧基线 0.15–0.23、踏步缺陷 19.4）==="
    column -t -s$'\t' "$MARCH_TSV" | tee -a "$OUT/live_status.txt"; }
}

log "live sweep started: stride=${STRIDE} iters, poll=${POLL_S}s, arms='${ARMS}'"

round=0
while :; do
  round=$((round + 1))
  found=0
  for arm in $ARMS; do
    run="logs/rsl_rl/velocity/$arm"
    [ -d "$run" ] || continue
    while read -r ck; do
      [ -n "$ck" ] || continue
      it=$(basename "$ck" .pt | sed 's/model_//')
      label="${arm}_ck$(printf '%05d' "$it")"
      done_label "$label" && continue
      onnx="$OUT/${label}.onnx"
      if [ ! -f "$onnx" ]; then
        .venv/bin/python scripts/export.py "$TASK" --checkpoint-file "$ck" \
          --onnx-file "$onnx" > "$OUT/${label}_export.log" 2>&1 || {
            log "[$label] export failed (will retry next round)"; continue; }
      fi
      log "[$label] evaluating (iter ${it})"
      eval_one "$label" "$onnx" || continue
      measure_marching "$label" "$onnx" || true
      found=$((found + 1))
    done < <(ls "$run"/model_*.pt 2>/dev/null | sort -t_ -k2 -n |
               awk -F'model_' -v s="$STRIDE" '{n=$2+0; if (n > 0 && n % s == 0) print}')
  done
  [ "$found" -gt 0 ] && report_so_far

  # Stop once training is over AND every stride checkpoint has been measured.
  if ! pgrep -f "mjlab_microduck.train_cli" >/dev/null; then
    pending=0
    for arm in $ARMS; do
      while read -r ck; do
        [ -n "$ck" ] || continue
        it=$(basename "$ck" .pt | sed 's/model_//')
        done_label "${arm}_ck$(printf '%05d' "$it")" || pending=$((pending + 1))
      done < <(ls "logs/rsl_rl/velocity/$arm"/model_*.pt 2>/dev/null | sort -t_ -k2 -n |
                 awk -F'model_' -v s="$STRIDE" '{n=$2+0; if (n > 0 && n % s == 0) print}')
    done
    if [ "$pending" -eq 0 ]; then
      log "training finished and all stride checkpoints measured; live sweep exiting"
      break
    fi
    log "training finished; ${pending} stride checkpoints still pending (model files may still be flushing)"
  fi
  sleep "$POLL_S"
done

report_so_far
log "live sweep complete"
