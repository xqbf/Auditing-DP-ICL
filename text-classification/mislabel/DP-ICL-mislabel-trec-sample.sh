#!/bin/bash
#SBATCH --job-name=yxia327-test
#SBATCH --output=slurm_logs/test-run-%j.out
#SBATCH --gres=gpu:1
#SBATCH --mem=2GB
#SBATCH --ntasks=1                   # 运行4个MPI进程
#SBATCH --cpus-per-task=2            # 每个进程使用2个CPU核心进行多线程计算

cd /local/scratch/yxia327/mia/MIA-ICL/text-classification/mislabel
source /local/scratch/yxia327/venv/bin/activate

# --- Modificación principal ---
# Pasar ambos argumentos, RUN_ID y NUM_SUBSETS, al script de Python
python DP-ICL-mislabel-trec-sample.py 
