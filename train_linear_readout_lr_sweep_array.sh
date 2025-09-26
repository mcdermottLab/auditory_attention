#!/bin/bash -l
#SBATCH --job-name=linread_lr_sweep
#SBATCH --output=outLogs/lr_sweep_%A_%a.out
#SBATCH --error=outLogs/lr_sweep_%A_%a.err
#SBATCH --partition=mcdermott
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=10
#SBATCH --mem=500G
#SBATCH --time=1-00:00:00
#SBATCH --array=0               # 7 layers (0-6) × 5 learning rates = 35 jobs

source /etc/profile.d/modules.sh
source /om2/user/rphess/miniforge3/etc/profile.d/conda.sh

export HDF5_USE_FILE_LOCKING=FALSE

conda activate /om2/user/imgriff/conda_envs/pytorch_2

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64

# Define learning rates in log space from 1e-6 to 1e-4
LEARNING_RATES=(1e-8 3.16e-8 1e-7 3.16e-7 1e-6)

# Calculate layer index and learning rate index from array task ID
LAYER_IDX=$((SLURM_ARRAY_TASK_ID / 5))
LR_IDX=$((SLURM_ARRAY_TASK_ID % 5))

# Get the learning rate for this job
LR=${LEARNING_RATES[$LR_IDX]}

echo "Running layer $LAYER_IDX with learning rate $LR"

# Run training for the given layer index and learning rate
torchrun --standalone --nproc_per_node=2 \
  train_linear_readout_lr_sweep.py \
  --layer_idx "$LAYER_IDX" \
  --lr_word "$LR" \
  --lr_azim "$LR" \
  --lr_f0 "$LR" \
  --tasks "num_azim_classes" \
  --config_path "config/linear_readout/linear_readout_layer_${LAYER_IDX}.yaml" \
  --save_outputs
