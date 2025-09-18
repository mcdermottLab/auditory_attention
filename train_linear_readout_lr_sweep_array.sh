#!/bin/bash -l
#SBATCH --job-name=linread_lr_sweep
#SBATCH --output=outLogs/lr_sweep_%A_%a.out
#SBATCH --error=outLogs/lr_sweep_%A_%a.err
#SBATCH --partition=mcdermott
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=20
#SBATCH --mem=160G
#SBATCH --time=1-00:00:00
#SBATCH --array=10-34                  # 7 layers (0-6) × 5 learning rates = 35 jobs

source /etc/profile.d/modules.sh
source /om2/user/rphess/miniforge3/etc/profile.d/conda.sh

export HDF5_USE_FILE_LOCKING=FALSE

conda activate /om2/user/imgriff/conda_envs/pytorch_2

# Define learning rates in log space from 1e-5 to 1e-3
LEARNING_RATES=(1e-5 3.16e-5 1e-4 3.16e-4 1e-3)

# Calculate layer index and learning rate index from array task ID
LAYER_IDX=$((SLURM_ARRAY_TASK_ID / 5))
LR_IDX=$((SLURM_ARRAY_TASK_ID % 5))

# Get the learning rate for this job
LR=${LEARNING_RATES[$LR_IDX]}

echo "Running layer $LAYER_IDX with learning rate $LR"

# Run training for the given layer index and learning rate
srun python train_linear_readout_lr_sweep.py \
  --layer_idx "$LAYER_IDX" \
  --lr_word "$LR" \
  --lr_azim "$LR" \
  --lr_f0 "$LR" \
  --config_path "config/linear_readout/linear_readout_layer_${LAYER_IDX}.yaml" \
  --save_outputs
