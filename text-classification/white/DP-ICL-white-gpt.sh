#!/bin/bash
#SBATCH --job-name=vllm-exp
#SBATCH --gres=gpu:1                  # <--- 修改 1: 这里改成你需要的 GPU 数量 (1)
#SBATCH --mem=64GB
#SBATCH --output=slurm_logs/vllm_exp_%j.out
#SBATCH --error=slurm_logs/vllm_exp_%j.err

# === 环境准备 ===
# module load cuda/12.1             # 如果集群要手动 load 模块
source /local/scratch/yxia327/venv/bin/activate   # 替换成你的虚拟环境路径

# === 路径 ===
MODEL_PATH=/local/scratch/yxia327/gpt-oss-20b
CODE_PATH=/local/scratch/yxia327/mia/MIA-ICL/text-classification/white
LOG_DIR=$CODE_PATH/logs # 定义日志目录

# === 准备工作 (自动创建日志目录) ===
# 确保日志目录存在，避免 "No such file or directory" 错误
mkdir -p $LOG_DIR
mkdir -p slurm_logs

# === 启动 vLLM API 服务（后台运行） ===
echo "Starting vLLM server..."

export NCCL_P2P_DISABLE=1

python -u -m vllm.entrypoints.openai.api_server \
  --model $MODEL_PATH \
  --dtype bfloat16 \
  --tensor-parallel-size 1 \
  --port 8000 \
  --host 0.0.0.0 > "$LOG_DIR/vllm_server_$SLURM_JOB_ID.log" 2>&1 &

VLLM_PID=$!

# === 等待 server 启动稳定 ===
# 对于 20B 这样的大模型，加载需要时间，增加等待时间以确保服务完全启动
echo "Waiting for vLLM server to initialize... (PID: $VLLM_PID)"
sleep 120   # 建议延长等待时间到 2 分钟


cd $CODE_PATH
echo "Running experiment..."
python -u DP-ICL-white-gpt.py

# === 实验结束，关掉 vLLM ===
echo "Killing vLLM server (PID: $VLLM_PID)..."
kill $VLLM_PID
wait $VLLM_PID 2>/dev/null # 等待进程结束
echo "Server killed."