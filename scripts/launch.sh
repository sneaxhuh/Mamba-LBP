#!/bin/bash
set -euo pipefail

# Usage:
#   ./scripts/launch.sh] <node_rank> <master_node_number> [train_cmd ...args]
# Example (3 nodes):
#   # node0
#   ./launch.sh 0 1 python -m glmamba.train_lightning --data-root ... --out-dir ... --amp
#   # node1
#   ./launch.sh 1 1 python -m glmamba.train_lightning --data-root ... --out-dir ... --amp
#   # node2
#   ./launch.sh 2 1 python -m glmamba.train_lightning --data-root ... --out-dir ... --amp

# Node rank (0 for the first node, 1 for the second node, etc.)
NODE_RANK="$1"
MASTER_NODE_NUM="$2"

if [ -z "$NODE_RANK" ] || [ -z "$MASTER_NODE_NUM" ]; then
  echo "Usage: $0 <node_rank> <master_node_number> [train_script ...args]" >&2
  exit 2
fi

# Master node address:
# - If you pass master_node_number=1 and your nodes are named gpu001/gpu002/... this will work.
# - If running under Slurm, we prefer the first hostname from SLURM_NODELIST when available.
if [ -n "${SLURM_NODELIST:-}" ] && command -v scontrol >/dev/null 2>&1; then
  MASTER_ADDR="$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)"
else
  MASTER_ADDR="$(printf "gpu%03d" "$MASTER_NODE_NUM")"
fi

# Master port
MASTER_PORT=29500

# Training command (can be overridden by passing it after the first 2 args)
if [ "$#" -ge 3 ]; then
  shift 2
  TRAIN_CMD=("$@")
else
  TRAIN_CMD=(python -m glmamba.train_lightning)
fi

# Load toolchain modules (matching the environment used to build mamba-ssm)
module load gcc/11.1.0
module load cuda/11.6

# Optional: activate a conda env (set CONDA_ENV=glmamba, etc.)
if [ -n "${CONDA_ENV:-}" ]; then
  if command -v conda >/dev/null 2>&1; then
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"
  else
    echo "CONDA_ENV is set but 'conda' is not on PATH" >&2
    exit 2
  fi
fi

# Set the number of nodes and GPUs per node.
# Prefer Slurm-provided values when running under a job allocation.
TOTAL_NODES="${SLURM_NNODES:-3}"
NUM_GPUS="${SLURM_GPUS_ON_NODE:-2}"

# Enable detailed error reporting
export TORCHELASTIC_ERROR_FILE="/tmp/torch_elastic_error.log"

# Run the training script
torchrun \
    --nnodes=$TOTAL_NODES \
    --nproc_per_node=$NUM_GPUS \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    "${TRAIN_CMD[@]}"