#!/bin/bash -l
#SBATCH --job-name=linread_layers
#SBATCH --output=outLogs/linread_%A_%a.out
#SBATCH --error=outLogs/linread_%A_%a.err
#SBATCH --partition=mcdermott
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=20
#SBATCH --mem=160G
#SBATCH --time=2-00:00:00
#SBATCH --array=2                  # ex: run layers 0..6 (adjust as needed)

source /etc/profile.d/modules.sh
source /om2/user/rphess/miniforge3/etc/profile.d/conda.sh

export HDF5_USE_FILE_LOCKING=FALSE

conda activate /om2/user/imgriff/conda_envs/pytorch_2

# Run training for the given layer index
LAYER_IDX="${SLURM_ARRAY_TASK_ID}"

srun python train_linear_readout_by_layer.py \
  --layer_idx "$LAYER_IDX" \
  --config_path "config/linear_readout/linear_readout_layer_${LAYER_IDX}.yaml" \
  --tasks "num_word_classes" \
  --save_outputs