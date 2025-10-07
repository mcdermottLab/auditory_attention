#!/bin/bash -l
#SBATCH --job-name=linread_val
#SBATCH --output=outLogs/linread_val_%A_%a.out
#SBATCH --error=outLogs/linread_val_%A_%a.err
#SBATCH --mem=180G
#SBATCH -N 1
#SBATCH --cpus-per-task=20
#SBATCH --ntasks-per-node=1
#SBATCH --time=1-00:00:00
#SBATCH --partition=mcdermott
#SBATCH --gres=gpu:a100:1
#SBATCH --array=5-6               # ex: run layers 0..6 (adjust as needed)

source /etc/profile.d/modules.sh
source /om2/user/rphess/miniforge3/etc/profile.d/conda.sh

export HDF5_USE_FILE_LOCKING=FALSE

conda activate /om2/user/imgriff/conda_envs/pytorch_2

# Run training for the given layer index
LAYER_IDX="${SLURM_ARRAY_TASK_ID}"

srun python linear_readout_val.py \
  --layer_idx "$LAYER_IDX" \
  --config_path "config/linear_readout/linear_readout_layer_${LAYER_IDX}.yaml" \
  --task "num_azim_classes"  \
  --num_gpus 1 \
  --save_outputs \
