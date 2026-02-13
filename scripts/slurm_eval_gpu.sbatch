#!/usr/bin/env bash
#SBATCH --job-name=glmamba-eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH -A iitr
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x.%j.out
#SBATCH --error=logs/%x.%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

PYTHON="./.venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then
  PYTHON="python3"
fi

# Required:
#   --data-root: BraTS2021 root
#   --test-subjects: text file with subject IDs
#   --checkpoint: path to best_psnr.pt or last.pt

srun "$PYTHON" -m glmamba.eval \
  --data-root "/scratch/$USER/brats2021" \
  --test-subjects "/scratch/$USER/glmamba_runs/run_001/val_subjects.txt" \
  --scale 2 \
  --checkpoint "/scratch/$USER/glmamba_runs/run_001/best_psnr.pt"

