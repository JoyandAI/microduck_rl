#!/usr/bin/env bash
# =============================================================================
#  microduck_rl — AutoDL dependency installer (third-party libs only; the repo
#  itself is installed separately, see the trailing notes).
#
#  The dependency list lives in requirements-autodl.txt. This script reads it
#  line by line and constrains each install to the versions in uv.lock.
#
#  Usage (repo already uploaded to the instance):
#      cd /root/autodl-tmp/microduck_rl
#      bash scripts/autodl_setup.sh                # install uv and dependencies using the selected index
#      bash scripts/autodl_setup.sh --upstream     # use upstream PyPI (mirror missing packages)
#      bash scripts/autodl_setup.sh --list         # print the plan, install nothing
#      bash scripts/autodl_setup.sh --train --task Mjlab-StandUp-Flat-OpenMD
#
#  Target environment: Ubuntu + NVIDIA driver, RTX 5090 (Blackwell sm_120):
#    - torch 2.9.1 PyPI x86_64 wheel bundles CUDA 12.8 runtime and carries sm_120
#    - warp-lang 1.12.0 bundles CUDA toolkit 12.9; NVRTC supports arch 120/121
#    => pick a CUDA 12.8+ image (driver >= 570); no system CUDA toolkit needed
# =============================================================================

if [ -z "${BASH_VERSION:-}" ]; then echo "run this with bash: bash scripts/autodl_setup.sh" >&2; exit 1; fi
set -euo pipefail

MIRROR=""
UPSTREAM=0
DO_LIST=0
DO_TRAIN=0
TASK="Mjlab-StandUp-Flat-OpenMD"
ENVS=4096
ITERS=""

while [ $# -gt 0 ]; do
  case "$1" in
    --upstream) UPSTREAM=1; shift ;;
    --index)    MIRROR="${2:?--index needs a URL}"; shift 2 ;;
    --list)     DO_LIST=1; shift ;;
    --train)    DO_TRAIN=1; shift ;;
    --task)     TASK="${2:?}"; shift 2 ;;
    --envs)     ENVS="${2:?}"; shift 2 ;;
    --iters)    ITERS="${2:?}"; shift 2 ;;
    -h|--help)  sed -n '2,20p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 1 ;;
  esac
done

# Which PyPI index to use, in order of precedence:
#   --upstream  >  --index URL  >  UV_DEFAULT_INDEX / PIP_INDEX_URL  >
#   an index in pip.conf, if present  >  Tsinghua
if [ "$UPSTREAM" = "1" ]; then
  MIRROR="https://pypi.org/simple"
elif [ -n "$MIRROR" ]; then
  :                                                     # --index wins
elif [ -n "${UV_DEFAULT_INDEX:-}" ]; then
  MIRROR="$UV_DEFAULT_INDEX"
elif [ -n "${PIP_INDEX_URL:-}" ]; then
  MIRROR="$PIP_INDEX_URL"
else
  for conf in /etc/pip.conf "$HOME/.config/pip/pip.conf" "$HOME/.pip/pip.conf"; do
    [ -f "$conf" ] || continue
    MIRROR="$(awk -F'=' '/^[[:space:]]*index-url/{gsub(/[[:space:]]/, "", $2); print $2; exit}' "$conf")"
    [ -n "$MIRROR" ] && { echo "index: $MIRROR (from $conf)"; break; }
  done
  [ -n "$MIRROR" ] || MIRROR="https://pypi.tuna.tsinghua.edu.cn/simple"
fi

# Huge CUDA wheels (nvidia-cublas / cudnn / cufft ...) are what make this install slow,
# and mirrors tend to throttle single long transfers: let uv fetch several in parallel.
export UV_CONCURRENT_DOWNLOADS="${UV_CONCURRENT_DOWNLOADS:-16}"
UV_CACHE_GIVEN="${UV_CACHE_DIR:+1}"     # a caller-provided cache dir wins over the data-disk default

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
echo "repo root: $REPO_ROOT"

REQ="$REPO_ROOT/requirements-autodl.txt"
[ -f "$REQ" ]     || { echo "missing requirements-autodl.txt (dependency list) — incomplete upload?" >&2; exit 1; }
[ -f uv.lock ]    || { echo "missing uv.lock — upload it with the training source" >&2; exit 1; }
[ -d vendor/bam ] || { echo "missing vendor/bam (BAM actuator models, must be uploaded too)" >&2; exit 1; }
printf -v EDITABLE_CMD 'uv pip install --python .venv/bin/python --index-url %q --no-deps -e .' "$MIRROR"

# Parse the list: drop comments and blank lines.
mapfile -t PKGS < <(sed 's/#.*//' "$REQ" | sed 's/[[:space:]]*$//' | grep -v '^[[:space:]]*$')

if [ "$DO_LIST" = "1" ]; then
  echo
  echo "selected task: $TASK"
  echo "will install these packages in order, constrained by uv.lock (index: $MIRROR):"
  i=0
  for p in "${PKGS[@]}"; do
    i=$((i+1))
    printf '  %2d. uv pip install %s\n' "$i" "$p"
  done
  echo
  echo "then install the training package: $EDITABLE_CMD"
  exit 0
fi

# --- keep caches and the venv on the data disk (AutoDL system disk is often 30 GB) ---
if [ -d /root/autodl-tmp ] && [ -w /root/autodl-tmp ]; then
  if [ -z "$UV_CACHE_GIVEN" ]; then
    UV_CACHE_DIR="/root/autodl-tmp/.microduck/uv-cache"
  fi
  export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-/root/autodl-tmp/.microduck/uv-python}"
  mkdir -p "$UV_CACHE_DIR" "$UV_PYTHON_INSTALL_DIR"
  echo "uv cache -> $UV_CACHE_DIR"
fi
export UV_CACHE_DIR="${UV_CACHE_DIR:-$HOME/.cache/uv}"
export UV_HTTP_TIMEOUT=600          # the 30 s default always dies on 2 GB CUDA wheels

# --- uv ---
# The uv wheel and the standalone installer both provide the uv binary.
if ! command -v uv >/dev/null 2>&1; then
  echo "==> installing uv from the PyPI mirror ($MIRROR)"
  if command -v pip >/dev/null 2>&1; then
    pip install -i "$MIRROR" uv || true
    export PATH="$HOME/.local/bin:$PATH"
    hash -r
  fi
fi
if ! command -v uv >/dev/null 2>&1; then
  echo "==> mirror install missed uv; falling back to the official installer (GitHub)"
  echo "    this requires access to astral.sh and its release downloads"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
  hash -r
fi
command -v uv >/dev/null 2>&1 || { echo "could not install uv. Try: pip install -i $MIRROR uv" >&2; exit 1; }
echo "uv $(uv --version | awk '{print $2}')"

# --- venv (Python 3.12; pyproject pins >=3.12,<3.13) ---
echo
echo "==> creating .venv (Python 3.12)"
# Prefer an existing Python 3.12 on PATH; otherwise uv can download its managed build.
if [ -z "${UV_PYTHON_PREFERENCE:-}" ]; then
  for cand in python3.12 python3 python; do
    if command -v "$cand" >/dev/null 2>&1 && \
       "$cand" -c 'import sys; sys.exit(0 if sys.version_info[:2] == (3, 12) else 1)' >/dev/null 2>&1; then
      export UV_PYTHON_PREFERENCE=system
      echo "python: reusing the existing 3.12 at $(command -v "$cand") (no managed download)"
      break
    fi
  done
fi
if ! uv venv --python 3.12 --allow-existing .venv; then
  cat >&2 <<'HINT'
uv could not create the Python 3.12 environment. Check the error above.
If the interpreter download failed, make the Python release downloads reachable
from this shell or add an existing Python 3.12 to PATH, then rerun the same command.
HINT
  exit 1
fi

PYTHON="$REPO_ROOT/.venv/bin/python"
CONSTRAINTS=$(mktemp)
trap 'rm -f -- "$CONSTRAINTS"' EXIT
uv export --frozen --no-dev --no-hashes --no-emit-local --format requirements.txt > "$CONSTRAINTS"
PIP=(uv pip install --python "$PYTHON" --index-url "$MIRROR" --constraint "$CONSTRAINTS")

# --- install one entry at a time ---
TOTAL=${#PKGS[@]}
i=0
for p in "${PKGS[@]}"; do
  i=$((i+1))
  echo
  echo "=== [$i/$TOTAL] $p"
  read -r -a args <<< "$p"   # The editable entry has two arguments: -e ./vendor/bam.
  "${PIP[@]}" "${args[@]}"
done

echo
echo "==> checking the simulation stack together against uv.lock"
"${PIP[@]}" torch==2.9.1 warp-lang==1.12.0 mujoco==3.10.0 mujoco-warp==3.8.1 mjlab==1.3.0 wandb==0.24.0

# --- sanity check: is the GPU visible? ---
echo
echo "==> sanity check"
if ! "$PYTHON" - <<'PY'
from importlib.metadata import distributions, version

import mujoco
import mjlab
import onnxruntime
import torch
import warp

core = next(distributions(name="mujoco-core"), None)
if core is not None:
    raise SystemExit(f"mujoco-core {core.version} shares native-library paths with the locked mujoco {mujoco.__version__}")

print(f"mujoco {mujoco.__version__} | mjlab {version('mjlab')} | warp {warp.__version__} | onnxruntime {onnxruntime.__version__}")
print(f"torch {torch.__version__} | CUDA {torch.version.cuda} | devices {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"  cuda:{i} {p.name} {p.total_memory/1024**3:.1f} GiB sm_{p.major}{p.minor}")
if torch.cuda.device_count() == 0:
    print("  ! no GPU visible: check the driver (RTX 5090 needs CUDA 12.8+ / driver 570+) or the no-GPU mode")
PY
then
  echo "Dependency check failed. If this .venv was used for another install, preserve it and rerun in a fresh environment:" >&2
  echo '  mv -- .venv ".venv.before-repair-$(date +%Y%m%d_%H%M%S)"' >&2
  printf '  bash scripts/autodl_setup.sh --index %q --task %q\n' "$MIRROR" "$TASK" >&2
  exit 1
fi

# --- is the repo itself installed? (you install it; this only reminds and gates --train) ---
HAS_PACKAGE=$("$PYTHON" -c 'from importlib.util import find_spec; print(int(find_spec("mjlab_microduck") is not None))')
if [ "$HAS_PACKAGE" = 1 ]; then
  "$PYTHON" scripts/check_openmd_training.py --registered-task "$TASK"
else
  echo
  echo "NOTE: the repo itself is not installed in the venv yet. Install it with:"
  echo "    $EDITABLE_CMD"
  echo "  (--no-deps matters: otherwise the PyPI copy of better-actuator-models overwrites vendor/bam)"
  if [ "$DO_TRAIN" = "1" ]; then exit 1; fi
fi

# --- optional: launch training right away ---
echo
if [ "$DO_TRAIN" = "1" ]; then
  STAMP="$(date +%Y%m%d_%H%M%S)"
  mkdir -p logs/autodl
  LOG="$REPO_ROOT/logs/autodl/${STAMP}_${TASK}.log"
  ARGS=(train "$TASK" --env.scene.num-envs "$ENVS" --agent.logger tensorboard --agent.seed 42 --agent.run-name "autodl-$STAMP")
  [ -n "$ITERS" ] && ARGS+=(--agent.max_iterations "$ITERS")
  # setsid detaches the run from this terminal (survives SSH disconnects) and gives it its
  # own process group, so "kill -- -PID" stops uv and the trainer together.
  setsid uv run --no-sync "${ARGS[@]}" > "$LOG" 2>&1 < /dev/null &
  sleep 20
  if kill -0 $! 2>/dev/null; then
    echo "training started in background: PID $!"
    echo "  log:  tail -f $LOG"
    echo "  stop: kill -- -$!   (or pkill -f \"train $TASK\")"
  else
    tail -30 "$LOG"
    echo "training died on startup, log above" >&2
    exit 1
  fi
else
  cat <<EOF
done. next steps (keep --no-sync so uv does not re-install everything from uv.lock):

  # the repo itself (if not installed yet)
  $EDITABLE_CMD

  # smoke test (1-2 min, always run this first)
  uv run --no-sync train $TASK --env.scene.num-envs 64 --agent.max_iterations 5 --agent.logger tensorboard --agent.seed 42

  # full training
  bash scripts/autodl_setup.sh --train --task "$TASK" --envs $ENVS
  # or in the foreground (activate the venv and --no-sync is no longer needed)
  source .venv/bin/activate && train $TASK --env.scene.num-envs $ENVS --agent.logger tensorboard --agent.seed 42

  # variables to re-export in a fresh shell
EOF
  printf '  export UV_CACHE_DIR=%q UV_HTTP_TIMEOUT=%q\n' "$UV_CACHE_DIR" "$UV_HTTP_TIMEOUT"
fi
