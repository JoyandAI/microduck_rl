#!/usr/bin/env bash
# =============================================================================
# Q1 straightness matrix — "why does the HD-1910 policy veer and the official
# one not?" with enough environments to be evidence rather than a draw.
#
# 4 arms x 2 commands x 5 seeds, 64 envs each, 20 s, frozen commands, DR ON:
#   O_xl  official alpha_walking on ITS OWN hardware   (XL330 model + XL330 act)
#   O_hd  official policy on OUR hardware              (HD-1910 model + act)
#   U_hd  our walk_hd1910_12000 on OUR hardware
#   U_xl  our policy on the OFFICIAL hardware
#
# Why 64 envs: the repo's existing official-vs-local yaw numbers come from ONE
# env with seed 42 (docs/forensics/agent_B_official_reference.md §5 explicitly
# says they "should not be quoted without a multi-env/multi-seed repeat").
# With DR active each env is an independent draw, so 64 envs x 5 seeds = 320
# draws per cell.
#
# Inference is batched: the evaluator falls back to one `sess.run` per env per
# step for the fixed-batch-1 exports, which is ~50x slower. The dynamic-batch
# copies are produced + PROVEN equivalent (bit-identical) by
# scripts/make_dynamic_batch_onnx.py before any measurement.
#
# Resumable: a run whose JSON already exists is skipped.
#
# Usage:  bash scripts/q1_straightness_matrix.sh                # everything
#         ONLY=U_hd bash scripts/q1_straightness_matrix.sh      # one arm
#         SEEDS="0 1" VX="0.2" bash scripts/q1_straightness_matrix.sh
# =============================================================================
set -u
cd "$(dirname "$0")/.." || exit 1

export UV_CACHE_DIR=/tmp/uv-cache
export WANDB_MODE=offline

OUT=logs/q1_matrix
ONNXDIR="$OUT/onnx"
OURS="$ONNXDIR/ours_walk_dyn.onnx"
OFFICIAL="$ONNXDIR/official_walk_dyn.onnx"
SECONDS_PER_RUN=${SECONDS_PER_RUN:-20}
NUM_ENVS=${NUM_ENVS:-64}
SEEDS=${SEEDS:-"0 1 2 3 4"}
VX=${VX:-"0.0 0.2"}
ONLY=${ONLY:-""}
FREEZE=${FREEZE:-1}

mkdir -p "$OUT" "$ONNXDIR"

log() { echo "[$(date '+%F %T')] $*"; }

# ---- 1. build the dynamic-batch eval copies (verified, not assumed) --------
build_dyn() {  # $1 = src onnx, $2 = dst
  [ -f "$2" ] && { log "dynamic copy exists: $2"; return 0; }
  log "building dynamic-batch copy: $1 -> $2"
  uv run python scripts/make_dynamic_batch_onnx.py --onnx "$1" --write "$2" \
      --check-batch "$NUM_ENVS" || return 1
}

build_dyn walk_hd1910_12000.onnx "$OURS" || exit 1
build_dyn /home/joyandai/microduck/policies/alpha_walking.onnx "$OFFICIAL" || exit 1

# ---- 2. the matrix --------------------------------------------------------
run_cell() {  # $1=tag $2=onnx $3=model $4=actuator $5=vx $6=seed $7=drop_dr
  local tag=$1 onnx=$2 om=$3 oa=$4 vx=$5 seed=$6 dropdr=$7
  local dd="" sufx=""
  # NB: `[ "$dropdr" = "1" ]`, NOT ${dropdr:+...} — the latter fires for "0" too
  # (any non-empty value), which silently labels DR-ON runs as `_nodr`.
  if [ "$dropdr" = "1" ]; then dd="--drop-dr"; sufx="_nodr"; fi
  local js="$OUT/${tag}_vx${vx}_s${seed}${sufx}.json"
  local lg="$OUT/${tag}_vx${vx}_s${seed}${sufx}.log"
  if [ -f "$js" ]; then log "skip (done): $js"; return 0; fi
  log "run  $tag  vx=$vx seed=$seed model=$om act=$oa ${dd:-DR-on}"
  local frz=""; [ "$FREEZE" = "1" ] && frz="--freeze-pose-cmd"
  uv run python scripts/eval_straightness.py \
      --onnx "$onnx" --lin-vel-x "$vx" --lin-vel-y 0.0 --ang-vel-z 0.0 \
      --num-envs "$NUM_ENVS" --seed "$seed" --seconds "$SECONDS_PER_RUN" \
      --override-model "$om" --override-actuator "$oa" $frz $dd \
      --json "$js" >"$lg" 2>&1
  local rc=$?
  if [ $rc -ne 0 ]; then
    log "FAILED rc=$rc  $tag vx=$vx s=$seed  (see $lg)"
    tail -5 "$lg"
  else
    log "ok   $tag vx=$vx s=$seed"
  fi
  return 0
}

for arm in O_xl O_hd U_hd U_xl; do
  case $arm in
    O_xl) onnx=$OFFICIAL; om=xl330;  oa=xl330  ;;
    O_hd) onnx=$OFFICIAL; om=hd1910; oa=hd1910 ;;
    U_hd) onnx=$OURS;     om=hd1910; oa=hd1910 ;;
    U_xl) onnx=$OURS;     om=xl330;  oa=xl330  ;;
  esac
  [ -n "$ONLY" ] && [ "$ONLY" != "$arm" ] && continue
  for vx in $VX; do
    for s in $SEEDS; do
      run_cell "$arm" "$onnx" "$om" "$oa" "$vx" "$s" 0
    done
  done
  # deterministic component (DR removed) at the walking command
  [ -n "$ONLY" ] && [ "$ONLY" != "$arm" ] && continue
  for s in 0 1; do
    run_cell "$arm" "$onnx" "$om" "$oa" 0.2 "$s" 1
  done
done

log "matrix complete -> $OUT"
log "aggregate with: uv run python scripts/q1_matrix_report.py --dir $OUT"
