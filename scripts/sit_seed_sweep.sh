#!/usr/bin/env bash
# =============================================================================
# Is the SIT half still there? Repeated, SEEDED draws of the same ONNX.
#
# Why this exists (2026-09-16): `eval_sitstand_head.py` was unseeded and sitstand
# DR is NOT gated on play=True, so every invocation draws its own CoM / encoder
# bias / mass / friction. The doc's Q2 entry said "fix verified at 3000: sit and
# rise preserved" from ONE draw that reached z=0.058 during the SIT segment --
# while another draw of the SAME /tmp/sit_headfix_3000.onnx only reached 0.104,
# and ck5000/ck6000 reached 0.080/0.094. "It sits" and "it cannot sit" cannot both
# be properties of one artifact; the difference is the draw.
#
# So: N seeds per checkpoint, report the distribution of the SIT segment's MINIMUM
# trunk z (target 0.060) and of the STAND metrics. A criterion is only PASS if the
# policy clears it on EVERY draw, not on the luckiest one.
#
# Usage:
#   setsid nohup bash scripts/sit_seed_sweep.sh > logs/sit_seed_sweep.out 2>&1 < /dev/null &
# =============================================================================
set -u
cd "$(dirname "$0")/.." || exit 1

export UV_CACHE_DIR=/tmp/uv-cache
OUT="${OUT:-logs/sit_seed_sweep}"
mkdir -p "$OUT"
SEEDS="${SEEDS:-0 1 2 3}"
MIN_FREE_MIB="${MIN_FREE_MIB:-1500}"

log() { echo "[$(date '+%F %T')] $*"; }

# onnx:tag pairs. Only checkpoints that were actually exported.
ARMS="${ARMS:-/tmp/sit_headfix_3000.onnx:ck3000 /tmp/sit_4000.onnx:ck4000 /tmp/sit_5000.onnx:ck5000 /tmp/sit_6000.onnx:ck6000}"

free_mib() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }

for arm in $ARMS; do
  onnx="${arm%%:*}"; tag="${arm##*:}"
  [ -f "$onnx" ] || { log "SKIP $tag (no $onnx)"; continue; }
  for s in $SEEDS; do
    txt="$OUT/${tag}_seed${s}.txt"
    [ -s "$txt" ] && { log "have $tag seed $s"; continue; }
    # Wait for headroom. A 1-env eval is small but the two training runs own the card.
    for _ in $(seq 1 60); do
      f=$(free_mib); [ "$f" -ge "$MIN_FREE_MIB" ] && break
      log "wait: free ${f} MiB < ${MIN_FREE_MIB}"; sleep 60
    done
    log "=== $tag seed $s (free $(free_mib) MiB) ==="
    .venv/bin/python scripts/eval_sitstand_head.py --onnx "$onnx" \
        --seed "$s" --sit-schedule "0:3,1:4,0:4" \
        > "$txt" 2>&1
    rc=$?
    if [ $rc -ne 0 ]; then log "  FAILED rc=$rc (see $txt)"; tail -5 "$txt"; continue; fi
    # Long-format summary, ONE ROW PER PHASE per draw: a single "the stand numbers"
    # row per draw silently dropped the second STAND segment, and the two STAND
    # segments answer different questions (seg0 = can it stand at all; seg2 = did it
    # get back UP after sitting). Also emit the SIT segment's extreme.
    python3 - "$txt" "$tag" "$s" >> "$OUT/summary.tsv" <<'PY'
import re, sys
p, tag, seed = sys.argv[1], sys.argv[2], sys.argv[3]
txt = open(p, encoding="utf-8", errors="replace").read()
# steady-state table row:
#   phase  z  tilt  neck_rel  head_rel  neck_abs  hcmd_neck  err  headF_N  head_z_mm
row = re.compile(r"^(seg\d+_(?:STAND|SIT))\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)"
                 r"\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s*$", re.M)
rows = list(row.finditer(txt))
if not rows:
    print(f"{tag}\t{seed}\tNOPARSE\t\t\t\t\t", file=sys.stdout)
for m in rows:
    ph, z, tilt, headf, headz = m.group(1), m.group(2), m.group(3), m.group(9), m.group(10)
    print(f"{tag}\t{seed}\t{ph}\t{z}\t{tilt}\t{headf}\t{headz}")
for m in re.finditer(r"(seg\d+_SIT)\s+z min\s+([-\d.]+)", txt):
    print(f"{tag}\t{seed}\t{m.group(1)}_MINZ\t{m.group(2)}\t\t\t")
# Time from the sit command to the first z <= 0.065 ("prompt" vs "hesitant").
# Absent in logs written before this metric existed -> emit nothing, not a lie.
for m in re.finditer(r"(seg\d+_SIT)\s+z min\s+[-\d.]+\s+max\s+[-\d.]+\s+\(target [\d.]+\)"
                     r"\s+reached [\d.]+ at t=([\d.]+) s", txt):
    print(f"{tag}\t{seed}\t{m.group(1)}_SITT\t{m.group(2)}\t\t\t")
for m in re.finditer(r"(seg\d+_SIT)\s+z min\s+[-\d.]+\s+max\s+[-\d.]+\s+\(target [\d.]+\)"
                     r"\s+NEVER reached", txt):
    print(f"{tag}\t{seed}\t{m.group(1)}_SITT\tNEVER\t\t\t")
PY
    log "  -> $(grep -c . "$OUT/summary.tsv" 2>/dev/null) rows so far"
  done
done

log "=== DONE ==="
printf '%-8s%6s%-16s%9s%8s%10s%11s\n' arm seed phase z_m tilt headF_N head_z_mm
sort -k1,1 -k2,2n -k3,3 "$OUT/summary.tsv" 2>/dev/null | \
  awk -F'\t' '{printf "%-8s%6s%-16s%9s%8s%10s%11s\n",$1,$2,$3,$4,$5,$6,$7}'
echo
echo "--- SIT segment minimum z per draw (target 0.060; PASS = every draw <= 0.065) ---"
awk -F'\t' '$3 ~ /_MINZ$/ {printf "%-8s seed %-3s  %s\n",$1,$2,$4}' "$OUT/summary.tsv" | sort
log "summary: $OUT/summary.tsv"
