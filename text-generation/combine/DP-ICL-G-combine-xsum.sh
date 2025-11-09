#!/bin/bash
#SBATCH --job-name=yxia327-test
#SBATCH --output=slurm_logs/test-run-%j.out
#SBATCH --gres=gpu:1
#SBATCH --mem=120GB


cd /local/scratch/yxia327/mia/MIA-ICL/text-generation/combine
source /local/scratch/yxia327/venv/bin/activate

# --- Modificación principal ---
# Pasar ambos argumentos, RUN_ID y NUM_SUBSETS, al script de Python
python DP-ICL-G-combine-xsum.py