#!/usr/bin/env bash
#SBATCH --job-name=glmamba-train
#SBATCH --partition=gpu
#SBATCH --account=iitr
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00
#SBATCH --output=logs/glmamba-train.%j.out
#SBATCH --error=logs/glmamba-train.%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

module purge
module load cuda/11.6
module load python/python3.10

source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate glmamba

srun python -m glmamba.train \
  --data-root "/scratch/$USER/brats2021" \
  --out-dir "/scratch/$USER/glmamba_runs/run_001" \
  --scale 2 \
  --epochs 50 \
  --batch-size 2 \
  --num-workers 8 \
  --lr 2e-4 \
  --amp
