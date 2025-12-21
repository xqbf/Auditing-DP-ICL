#!/bin/bash
#SBATCH --job-name=yxia327-test
#SBATCH --output=slurm_logs/test-run-%j.out
#SBATCH --gres=gpu:1
#SBATCH --mem=2GB
#SBATCH --ntasks=1                 
#SBATCH --cpus-per-task=2           

cd /local/scratch/yxia327/mia/MIA-ICL/text-classification/mislabel
source /local/scratch/yxia327/venv/bin/activate

python DP-ICL-mislabel-agnews.py 
