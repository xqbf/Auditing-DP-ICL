#!/bin/bash
#SBATCH --job-name=vllm-exp
#SBATCH --gres=gpu:1
#SBATCH --mem=64GB
#SBATCH --output=slurm_logs/vllm_exp_%j.out
#SBATCH --error=slurm_logs/vllm_exp_%j.err

set -euo pipefail

# =============================================================================
# Parameterized vLLM + experiment runner (Slurm)
# - SBATCH resources are NOT runtime-parameterizable; override via `sbatch --gres=... --mem=...`
# - Script args configure model paths/ports/etc and forward extra args to python after `--`.
# =============================================================================

# Defaults (match your original script)
VENV_ACTIVATE="/local/scratch/yxia327/venv/bin/activate"
MODEL_PATH="/local/scratch/yxia327/gpt-oss-20b"
CODE_PATH="/local/scratch/yxia327/mia/MIA-ICL/text-generation/combine"
HOST="0.0.0.0"
PORT="8000"
DTYPE="bfloat16"
TP="1"
WAIT_SEC="120"
PYTHON_EXE="python"
PY_SCRIPT="DP-ICL-G-vllm.py"   # default to argparse version
LOG_DIR=""   # if empty -> $CODE_PATH/logs
SLURM_LOG_DIR="slurm_logs"

# Parse args (bash-level)
PY_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --venv_activate) VENV_ACTIVATE="$2"; shift 2;;
    --model_path) MODEL_PATH="$2"; shift 2;;
    --code_path) CODE_PATH="$2"; shift 2;;
    --host) HOST="$2"; shift 2;;
    --port) PORT="$2"; shift 2;;
    --dtype) DTYPE="$2"; shift 2;;
    --tp) TP="$2"; shift 2;;
    --wait_sec) WAIT_SEC="$2"; shift 2;;
    --python_exe) PYTHON_EXE="$2"; shift 2;;
    --py_script) PY_SCRIPT="$2"; shift 2;;
    --log_dir) LOG_DIR="$2"; shift 2;;
    --slurm_log_dir) SLURM_LOG_DIR="$2"; shift 2;;
    --) shift; PY_ARGS+=("$@"); break;;
    -h|--help)
      cat <<'EOF'
Usage:
  sbatch DP-ICL-G-combine-gpt_param.sh [bash-args] [-- python-args...]

Bash args (configure vLLM / paths):
  --venv_activate PATH
  --model_path PATH
  --code_path PATH
  --host HOST
  --port PORT
  --dtype bfloat16|float16|float32
  --tp INT
  --wait_sec SECONDS
  --python_exe python|python3|...
  --py_script SCRIPTNAME (inside CODE_PATH)
  --log_dir PATH (default: CODE_PATH/logs)
  --slurm_log_dir PATH (default: slurm_logs)

Forward args to python after `--`, e.g.
  sbatch DP-ICL-G-combine-gpt_param.sh -- --data_path ... --subsets_list 4 8 --epsilons 1 2
EOF
      exit 0
      ;;
    *)
      echo "[ERROR] Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

if [[ -z "$LOG_DIR" ]]; then
  LOG_DIR="$CODE_PATH/logs"
fi

mkdir -p "$LOG_DIR"
mkdir -p "$SLURM_LOG_DIR"

# === Environment ===
# venv activate may reference unset env vars (e.g., LD_LIBRARY_PATH). Temporarily disable nounset.
set +u
source "$VENV_ACTIVATE"
set -u

export NCCL_P2P_DISABLE=1

echo "Starting vLLM server..."
echo "  MODEL_PATH=$MODEL_PATH"
echo "  HOST=$HOST PORT=$PORT DTYPE=$DTYPE TP=$TP"
echo "  LOG_DIR=$LOG_DIR"

$PYTHON_EXE -u -m vllm.entrypoints.openai.api_server   --model "$MODEL_PATH"   --dtype "$DTYPE"   --tensor-parallel-size "$TP"   --port "$PORT"   --host "$HOST" > "$LOG_DIR/vllm_server_${SLURM_JOB_ID}.log" 2>&1 &

VLLM_PID=$!

cleanup() {
  echo "Killing vLLM server (PID: $VLLM_PID)..."
  kill "$VLLM_PID" 2>/dev/null || true
  wait "$VLLM_PID" 2>/dev/null || true
  echo "Server killed."
}
trap cleanup EXIT INT TERM

echo "Waiting for vLLM server to initialize... (PID: $VLLM_PID)"
sleep "$WAIT_SEC"

cd "$CODE_PATH"
echo "Running experiment..."
echo "  PY_SCRIPT=$PY_SCRIPT"
echo "  PY_ARGS=${PY_ARGS[*]:-<none>}"

$PYTHON_EXE -u "$PY_SCRIPT" "${PY_ARGS[@]}"
