#!/bin/bash -l
#SBATCH --job-name=linread_layers
#SBATCH --output=outLogs/linread_%A_%a.out
#SBATCH --error=outLogs/linread_%A_%a.err
#SBATCH --partition=ou_bcs_normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=100G
#SBATCH --time=12:00:00
#SBATCH --array=5                  # ex: run layers 0..6 (adjust as needed)

# Load modules
module load miniforge
conda activate pytorch_2

# Project root and paths
PROJECT_ROOT="/orcd/data/jhm/001/om2/rphess/projects/github.com/Auditory-Attention"
export PYTHONPATH="${PYTHONPATH}:${PROJECT_ROOT}"
cd "$PROJECT_ROOT"

# Set up environment variables for distributed training (if needed)
export MASTER_ADDR=localhost
export MASTER_PORT=29500

# Print environment info for debugging
echo "=== Environment Setup ==="
echo "Array task id: ${SLURM_ARRAY_TASK_ID}"
echo "Python path: $PYTHONPATH"
echo "Conda environment: $CONDA_DEFAULT_ENV"
echo "CUDA visible devices: $CUDA_VISIBLE_DEVICES"
echo "GPUs requested: ${SLURM_GPUS_ON_NODE}"
echo "CPUs per task: ${SLURM_CPUS_PER_TASK}"
echo "Working directory: $(pwd)"
echo "========================="

# Run training for the given layer index
LAYER_IDX="${SLURM_ARRAY_TASK_ID}"

srun python train_linear_readout_by_layer.py \
  --layer_idx "$LAYER_IDX" \
  --config_path "config/linear_readout/linear_readout_layer_${LAYER_IDX}.yaml"