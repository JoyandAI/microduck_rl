#!/usr/bin/env bash
# Wait for the posture-fix walking run to finish, then export its final
# checkpoint to ONNX and run the 2026-09-11 posture acceptance table.
#
# Why a chain instead of two manual steps: the training is ~5.5 h, and the
# acceptance numbers are the ONLY thing that answers the user's actual question
# ("does it now look like the official one?"). Leaving the export+check to be
# remembered later loses the answer; this makes it automatic.
#
# Usage:
#   scripts/after_training_acceptance.sh <run_dir> [log_file]
# e.g.
#   scripts/after_training_acceptance.sh \
#       logs/rsl_rl/velocity/2026-09-11_13-09-08_velocity \
#       train_velocity_posture_fix.log
#
# It polls for the last checkpoint, waits until BOTH the checkpoint exists and
# the training process has exited, then exports and evaluates.

set -u

RUN_DIR="${1:?usage: $0 <run_dir> [log_file]}"
LOG_FILE="${2:-}"
POLL_S=60
ONNX_OUT="walk_posture_fix.onnx"
OUT_MD="docs/posture_fix_acceptance_result.md"

cd "$(dirname "$0")/.." || exit 1
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}"

# The final checkpoint is DISCOVERED, not hardcoded. A resumed run gets
# `max_iterations` ADDED to the iteration it loaded, so the last file is
# model_18499.pt rather than model_14999.pt after resuming from 3500/15000.
# Hardcoding model_14999.pt once made this chain wait forever on a file that
# was never going to appear.
last_ckpt() {
  ls "$RUN_DIR"/model_*.pt 2>/dev/null \
    | sed 's/.*model_//;s/\.pt$//' | grep -E '^[0-9]+$' | sort -n | tail -1
}

echo "[chain] waiting for training to exit (run_dir=$RUN_DIR, poll ${POLL_S}s)"

while true; do
  ck="$(last_ckpt)"
  running=$(pgrep -fc "train Mjlab-Velocity-Flat-MicroDuck" 2>/dev/null || echo 0)

  # Done when a checkpoint exists AND no trainer process is left. Keying off the
  # PROCESS instead of a fixed iteration string is what makes this resume-proof.
  if [ -n "${ck:-}" ] && [ "$running" = "0" ]; then
    echo "[chain] training finished (last checkpoint = model_${ck}.pt)"
    break
  fi
  sleep "$POLL_S"
done

CKPT="${RUN_DIR}/model_$(last_ckpt).pt"
echo "[chain] exporting ${CKPT} -> ${ONNX_OUT}"
uv run scripts/export.py Mjlab-Velocity-Flat-MicroDuck \
  --checkpoint-file "$CKPT" \
  --onnx-file "$ONNX_OUT" \
  --num-envs 1 \
  > /tmp/posture_export.log 2>&1
rc=$?
if [ "$rc" != "0" ] || [ ! -f "$ONNX_OUT" ]; then
  echo "[chain] EXPORT FAILED (rc=$rc); see /tmp/posture_export.log"
  tail -40 /tmp/posture_export.log
  exit 1
fi
echo "[chain] export ok: $(ls -la "$ONNX_OUT")"

echo "[chain] running the acceptance table"
uv run python scripts/check_posture_acceptance.py --onnx "$ONNX_OUT" --seconds 20 \
  > /tmp/posture_acceptance.log 2>&1
rc=$?
tail -60 /tmp/posture_acceptance.log

{
  echo "# Posture-fix acceptance result"
  echo
  echo "- run dir: \`${RUN_DIR}\`"
  echo "- onnx: \`${ONNX_OUT}\`"
  echo "- date: $(date -Is)"
  echo "- script exit: ${rc} (0 = all rows passed)"
  echo
  echo '```'
  sed -n '/POSTURE ACCEPTANCE/,$p' /tmp/posture_acceptance.log
  echo '```'
  echo
  echo "Criteria and the official/broken reference values:"
  echo "\`docs/posture_defect_analysis.md\` §7."
} > "$OUT_MD"

echo "[chain] wrote ${OUT_MD} (exit ${rc})"
exit "$rc"
