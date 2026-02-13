#!/usr/bin/env bash
#SBATCH --job-name=glmamba-train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH -A iitr
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x.%j.out
#SBATCH --error=logs/%x.%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

# Load Python module on compute node
module purge
module load python/python3.10

# Activate virtualenv (important!)
source .venv/bin/activate

srun python -m glmamba.train \
  --data-root "/scratch/$USER/brats2021" \
  --out-dir "/scratch/$USER/glmamba_runs/run_001" \
  --scale 2 \
  --epochs 50 \
  --batch-size 2 \
  --num-workers 4 \
  --lr 2e-4 \
  --amp
