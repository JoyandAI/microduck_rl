#!/usr/bin/env bash
# =============================================================================
# Overnight A/B queue — 2026-09-14
#
# Question: which of these is responsible for the HD-1910 walking policy veering
# (and for the huge run-to-run variance)?
#     A  our recipe as committed (22 reward terms, incl. heading_hold)
#     B  A + stiffer servo in sim (kp_fw 32 -> 64)
#     C  official recipe + heading_hold  (= our config minus our 3 "leg taxes" +
#        official pose/air_time implementations)
#     D  official recipe exactly (16 terms, origin/develop @ 3412ae5)
#
# 3 seeds per arm => 12 runs. The run-to-run variance measured on 2026-09-14 is
# as large as any effect we are chasing (same config, same seed, 3500 vs 12000
# iterations gave path efficiency 0.569 vs 0.124), so a single run per arm would
# be uninterpretable. Everything here is 3000 iterations at 2048 envs.
#
# Usage:  nohup setsid bash scripts/night_queue.sh > logs/night_queue.out 2>&1 &
# =============================================================================
set -u

cd "$(dirname "$0")/.." || exit 1
ROOT="$PWD"
OUT="$ROOT/logs/night_queue"
mkdir -p "$OUT"
RESULTS="$OUT/results.csv"
[ -f "$RESULTS" ] || echo "job,arm,seed,run_dir,onnx,yaw_bias_deg_s,path_efficiency,made_good_m,note" > "$RESULTS"

ITERS=${ITERS:-3000}
ENVS=${ENVS:-2048}
CONCURRENCY=${CONCURRENCY:-3}
EVAL_ENVS=${EVAL_ENVS:-64}

TASK_MAIN="Mjlab-Velocity-Flat-MicroDuck"
TASK_OFFICIAL="Mjlab-Velocity-Flat-MicroDuck-OfficialRecipe"

log() { echo "[$(date +%H:%M:%S)] $*"; }

# --- one training job --------------------------------------------------------
run_one() {
  local arm="$1" seed="$2"
  local job="${arm}_s${seed}"
  # UNIQUE per job: three jobs of the same arm run concurrently, so a shared
  # experiment_name would make the "newest run dir" lookup below pick the wrong
  # run (still-running sibling) and silently corrupt results.csv.
  local exp="night_${arm}_s${seed}"
  local jlog="$OUT/${job}.log"
  local task="$TASK_MAIN"
  local -a envs=()

  case "$arm" in
    A) envs=() ;;                                            # our recipe as-is
    B) envs=("MICRODUCK_KP_FW=64") ;;                        # stiffer servo (sim)
    C) envs=("MICRODUCK_STANCE_HYGIENE=0" \
             "MICRODUCK_PER_JOINT_POSTURE=0" \
             "MICRODUCK_AIR_TIME_GATE=0") ;;                 # official + heading_hold
    D) task="$TASK_OFFICIAL"; envs=() ;;                     # official recipe
    *) log "unknown arm $arm"; return 1 ;;
  esac

  log "START $job  task=$task  $([ ${#envs[@]} -gt 0 ] && printf '%s ' "${envs[@]}")"
  env WANDB_MODE=offline CUDA_VISIBLE_DEVICES=0 "${envs[@]}" \
      .venv/bin/python -m mjlab_microduck.train_cli "$task" \
      --env.scene.num-envs "$ENVS" \
      --agent.max_iterations "$ITERS" \
      --agent.seed "$seed" \
      --agent.experiment_name "$exp" \
      > "$jlog" 2>&1
  local rc=$?

  local rundir
  rundir=$(ls -dt "$ROOT/logs/rsl_rl/$exp"/*/ 2>/dev/null | head -1)
  local onnx=""
  [ -n "$rundir" ] && onnx=$(ls -t "$rundir"*.onnx 2>/dev/null | head -1)

  if [ $rc -ne 0 ]; then
    log "FAIL  $job rc=$rc  (see $jlog)"
    echo "$job,$arm,$seed,${rundir:-},${onnx:-},,,,train_rc=$rc" >> "$RESULTS"
    return 1
  fi
  log "DONE  $job  run=$rundir"

  # --- evaluate the final checkpoint (32 envs: fast, used for ranking) -------
  local ev="$OUT/${job}.json"
  CUDA_VISIBLE_DEVICES=0 timeout 1800 .venv/bin/python scripts/eval_straightness.py \
      --onnx "$onnx" --lin-vel-x 0.2 --seconds 20 --num-envs 32 --seed 0 \
      --json "$ev" --csv "$OUT/${job}.csv" > "$OUT/${job}.eval.log" 2>&1

  .venv/bin/python - "$ev" "$job" "$arm" "$seed" "$rundir" "$onnx" "$RESULTS" <<'PY'
import json, sys
ev, job, arm, seed, rundir, onnx, results = sys.argv[1:8]
try:
    d = json.load(open(ev)); a = d["aggregate"]
    yaw = a["yaw_rate_bias_deg_s"]["mean"]; eff = a["path_efficiency"]["mean"]
    mg = a["along_end_m"]["mean"]
    open(results, "a").write(
        f"{job},{arm},{seed},{rundir},{onnx},{yaw:+.3f},{eff:.4f},{mg:+.3f},\n")
except Exception as e:
    open(results, "a").write(f"{job},{arm},{seed},{rundir},{onnx},,,,eval_failed={e}\n")
PY
  log "EVAL  $job  $(tail -1 "$RESULTS")"
}

export -f run_one log
export ROOT OUT RESULTS ITERS ENVS TASK_MAIN TASK_OFFICIAL

# --- schedule: 4 arms x 3 seeds, N at a time --------------------------------
JOBS=()
for arm in A B C D; do
  for seed in 42 43 44; do JOBS+=("$arm $seed"); done
done

log "queue: ${#JOBS[@]} jobs, ${CONCURRENCY} concurrent, ${ITERS} iters @ ${ENVS} envs"
running=0
for j in "${JOBS[@]}"; do
  # throttle
  while [ "$(jobs -rp | wc -l)" -ge "$CONCURRENCY" ]; do sleep 20; done
  # GPU memory guard: need ~3 GiB per run
  free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
  while [ "${free_mib:-0}" -lt 3500 ]; do
    log "waiting for GPU memory (free=${free_mib} MiB)"
    sleep 60
    free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
  done
  run_one $j &
  sleep 45   # stagger startup (warp kernel cache + env build)
done
wait

log "ALL TRAINING DONE"
log "results:"
column -s, -t "$RESULTS" 2>/dev/null || cat "$RESULTS"
