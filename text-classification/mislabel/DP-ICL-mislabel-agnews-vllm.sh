#!/bin/bash
#SBATCH --job-name=vllm-exp
#SBATCH --gres=gpu:1
#SBATCH --mem=64GB
#SBATCH --output=slurm_logs/vllm_exp_%j.out
#SBATCH --error=slurm_logs/vllm_exp_%j.err

set -euo pipefail

# ============================================================
# Parameterized vLLM runner for AGNews mislabel experiments
# - SBATCH lines are parsed by Slurm before the script runs.
#   Override resources at submission time, e.g.:
#     sbatch --gres=gpu:2 --mem=128G this_script.sh ...
# ============================================================

usage() {
  cat << 'EOF'
Usage:
  sbatch DP-ICL-mislabel-agnews-gpt_param.sh [options] [-- python_args...]

Options:
  --venv_path PATH        Path to venv activate script
                          (default: /local/scratch/yxia327/venv/bin/activate)
  --model_path PATH       vLLM model path (default: /local/scratch/yxia327/gpt-oss-20b)
  --code_path PATH        Directory containing the experiment python file
                          (default: /local/scratch/yxia327/mia/MIA-ICL/text-classification/mislabel)
  --py_script FILE        Experiment python file to run (default: DP-ICL-mislabel-agnews-gpt.py)

  --log_dir PATH          Where to store vLLM server logs
                          (default: <code_path>/logs)
  --slurm_log_dir PATH    Where to store Slurm stdout/stderr (default: slurm_logs)

  --host HOST             vLLM host (default: 0.0.0.0)
  --port PORT             vLLM port (default: 8000)
  --dtype DTYPE           vLLM dtype (default: bfloat16)
  --tp N                  vLLM tensor-parallel-size (default: 1)
  --wait_sec SECONDS      Seconds to wait for vLLM to load (default: 120)

  --nccl_p2p_disable 0|1  Export NCCL_P2P_DISABLE (default: 1)
  -h, --help              Show this help

Notes:
  - Any arguments after a literal `--` are passed to the python experiment script.
    Example:
      sbatch DP-ICL-mislabel-agnews-gpt_param.sh -- --epsilons 1.0 --T_simulation 50000
EOF
}

# ---------------- defaults (match the original script) ----------------
VENV_PATH="/local/scratch/yxia327/venv/bin/activate"
MODEL_PATH="/local/scratch/yxia327/gpt-oss-20b"
CODE_PATH="/local/scratch/yxia327/mia/MIA-ICL/text-classification/mislabel"
PY_SCRIPT="DP-ICL-mislabel-agnews-vllm.py"

HOST="0.0.0.0"
PORT="8000"
DTYPE="bfloat16"
TP="1"
WAIT_SEC="120"

NCCL_P2P_DISABLE_VAL="1"

# ---------------- parse args ----------------
PY_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --venv_path) VENV_PATH="$2"; shift 2;;
    --model_path) MODEL_PATH="$2"; shift 2;;
    --code_path) CODE_PATH="$2"; shift 2;;
    --py_script) PY_SCRIPT="$2"; shift 2;;

    --log_dir) LOG_DIR="$2"; shift 2;;
    --slurm_log_dir) SLURM_LOG_DIR="$2"; shift 2;;

    --host) HOST="$2"; shift 2;;
    --port) PORT="$2"; shift 2;;
    --dtype) DTYPE="$2"; shift 2;;
    --tp) TP="$2"; shift 2;;
    --wait_sec) WAIT_SEC="$2"; shift 2;;

    --nccl_p2p_disable) NCCL_P2P_DISABLE_VAL="$2"; shift 2;;

    -h|--help) usage; exit 0;;
    --) shift; PY_ARGS+=("$@"); break;;
    *)
      echo "Unknown option: $1"
      echo
      usage
      exit 2
      ;;
  esac
done

# Resolve dependent defaults
LOG_DIR="${LOG_DIR:-$CODE_PATH/logs}"
SLURM_LOG_DIR="${SLURM_LOG_DIR:-slurm_logs}"

# ---------------- env & dirs ----------------
if [[ -f "$VENV_PATH" ]]; then
  # NOTE: This script uses `set -u` (nounset). Some virtualenv activation
  # scripts reference LD_LIBRARY_PATH even when it is unset, which triggers:
  #   /path/to/activate: line XX: LD_LIBRARY_PATH: unbound variable
  # Temporarily disable nounset while sourcing the activation script.
  set +u
  # shellcheck disable=SC1090
  source "$VENV_PATH"
  set -u
else
  echo "ERROR: venv activate script not found: $VENV_PATH"
  exit 1
fi

mkdir -p "$LOG_DIR"
mkdir -p "$SLURM_LOG_DIR"

export NCCL_P2P_DISABLE="$NCCL_P2P_DISABLE_VAL"

# ---------------- start vLLM ----------------
echo "Starting vLLM server..."
echo "  MODEL_PATH=$MODEL_PATH"
echo "  HOST=$HOST PORT=$PORT DTYPE=$DTYPE TP=$TP"
echo "  LOG_DIR=$LOG_DIR"
echo "  Experiment: $CODE_PATH/$PY_SCRIPT"
if [[ ${#PY_ARGS[@]} -gt 0 ]]; then
  echo "  Python args: ${PY_ARGS[*]}"
fi

python -u -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_PATH" \
  --dtype "$DTYPE" \
  --tensor-parallel-size "$TP" \
  --port "$PORT" \
  --host "$HOST" \
  > "$LOG_DIR/vllm_server_${SLURM_JOB_ID}.log" 2>&1 &

VLLM_PID=$!

cleanup() {
  echo "Killing vLLM server (PID: $VLLM_PID)..."
  kill "$VLLM_PID" 2>/dev/null || true
  wait "$VLLM_PID" 2>/dev/null || true
  echo "Server killed."
}
trap cleanup EXIT INT TERM

# Wait for the server to come up
echo "Waiting for vLLM server to initialize... (PID: $VLLM_PID) sleep=${WAIT_SEC}s"
sleep "$WAIT_SEC"

# ---------------- run experiment ----------------
cd "$CODE_PATH"
echo "Running experiment..."
python -u "$PY_SCRIPT" "${PY_ARGS[@]}"

echo "Experiment finished."
