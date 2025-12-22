#!/bin/bash
#SBATCH --job-name=vllm-exp
#SBATCH --gres=gpu:1
#SBATCH --mem=64GB
#SBATCH --output=slurm_logs/vllm_exp_%j.out
#SBATCH --error=slurm_logs/vllm_exp_%j.err

set -euo pipefail

# ------------------------------------------------------------------------------
# Parameter defaults (match original script behavior)
# ------------------------------------------------------------------------------
VENV_ACTIVATE="/local/scratch/yxia327/venv/bin/activate"
MODEL_PATH="/local/scratch/yxia327/gpt-oss-20b"
CODE_PATH="/local/scratch/yxia327/mia/MIA-ICL/text-classification/white"

# vLLM server
HOST="0.0.0.0"
PORT="8000"
DTYPE="bfloat16"
TP="1"
WAIT_SEC="120"

# logging
LOG_DIR=""            # if empty, will default to $CODE_PATH/logs
SLURM_LOG_DIR="slurm_logs"

# experiment runner
PYTHON_EXE="python"
PY_SCRIPT="DP-ICL-vllm.py"

# passthrough args to python after --
PY_ARGS=()

usage() {
  cat << EOF
Usage:
  sbatch $0 [options] [-- <python_args...>]

Options:
  --venv_activate PATH     Path to venv activate script (default: $VENV_ACTIVATE)
  --model_path PATH        vLLM model path (default: $MODEL_PATH)
  --code_path PATH         Working directory (default: $CODE_PATH)
  --host HOST              vLLM host (default: $HOST)
  --port PORT              vLLM port (default: $PORT)
  --dtype DTYPE            vLLM dtype (default: $DTYPE)
  --tp N                   tensor parallel size (default: $TP)
  --wait_sec SECONDS       sleep before running experiment (default: $WAIT_SEC)
  --log_dir PATH           vLLM log dir (default: \$CODE_PATH/logs)
  --slurm_log_dir PATH     slurm log dir (default: $SLURM_LOG_DIR)
  --python_exe CMD         python executable (default: $PYTHON_EXE)
  --py_script FILE         python script to run inside CODE_PATH (default: $PY_SCRIPT)

Examples:
  sbatch $0
  sbatch $0 --port 8001 --model_path /path/to/model --wait_sec 180
  sbatch $0 -- --data_path /path/to/data.csv --epsilons 1.0 2.0 --T_simulation 50000
EOF
}

# ------------------------------------------------------------------------------
# Parse args (supports: --key value ... and -- to end)
# ------------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --help|-h)
      usage; exit 0;;
    --venv_activate)
      VENV_ACTIVATE="$2"; shift 2;;
    --model_path)
      MODEL_PATH="$2"; shift 2;;
    --code_path)
      CODE_PATH="$2"; shift 2;;
    --host)
      HOST="$2"; shift 2;;
    --port)
      PORT="$2"; shift 2;;
    --dtype)
      DTYPE="$2"; shift 2;;
    --tp)
      TP="$2"; shift 2;;
    --wait_sec)
      WAIT_SEC="$2"; shift 2;;
    --log_dir)
      LOG_DIR="$2"; shift 2;;
    --slurm_log_dir)
      SLURM_LOG_DIR="$2"; shift 2;;
    --python_exe)
      PYTHON_EXE="$2"; shift 2;;
    --py_script)
      PY_SCRIPT="$2"; shift 2;;
    --)
      shift
      PY_ARGS+=("$@")
      break;;
    *)
      echo "Unknown option: $1"
      usage
      exit 1;;
  esac
done

if [[ -z "${LOG_DIR}" ]]; then
  LOG_DIR="${CODE_PATH}/logs"
fi

mkdir -p "${LOG_DIR}"
mkdir -p "${SLURM_LOG_DIR}"

# ------------------------------------------------------------------------------
# Environment
# ------------------------------------------------------------------------------
# IMPORTANT: venv activate may reference unset vars (e.g., LD_LIBRARY_PATH).
# Temporarily disable 'nounset' around 'source'.
set +u
source "${VENV_ACTIVATE}"
set -u

export NCCL_P2P_DISABLE=1

cleanup() {
  if [[ -n "${VLLM_PID:-}" ]]; then
    echo "Killing vLLM server (PID: ${VLLM_PID})..."
    kill "${VLLM_PID}" 2>/dev/null || true
    wait "${VLLM_PID}" 2>/dev/null || true
    echo "Server killed."
  fi
}
trap cleanup EXIT INT TERM

# ------------------------------------------------------------------------------
# Start vLLM server
# ------------------------------------------------------------------------------
echo "Starting vLLM server..."
${PYTHON_EXE} -u -m vllm.entrypoints.openai.api_server \
  --model "${MODEL_PATH}" \
  --dtype "${DTYPE}" \
  --tensor-parallel-size "${TP}" \
  --port "${PORT}" \
  --host "${HOST}" \
  > "${LOG_DIR}/vllm_server_${SLURM_JOB_ID}.log" 2>&1 &

VLLM_PID=$!
echo "Waiting for vLLM server to initialize... (PID: ${VLLM_PID})"
sleep "${WAIT_SEC}"

# ------------------------------------------------------------------------------
# Run experiment
# ------------------------------------------------------------------------------
cd "${CODE_PATH}"
echo "Running experiment: ${PYTHON_EXE} -u ${PY_SCRIPT} ${PY_ARGS[*]}"
${PYTHON_EXE} -u "${PY_SCRIPT}" "${PY_ARGS[@]}"
