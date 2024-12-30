#!/bin/bash
#SBATCH --job-name=train_binaural_attn
#SBATCH --output=outLogs/train_binaural_attn_v10_control_%A_%a.out
#SBATCH --error=outLogs/train_binaural_attn_v10_control_%A_%a.err

#SBATCH --mem=100GB
#SBATCH -N 1
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --partition=ou_bcs_low
#SBATCH --gres=gpu:h100:4
#SBATCH --array=0-2 # 0-2; 3 models in manifest

module load miniforge

export HDF5_USE_FILE_LOCKING=FALSE

mamba activate pytorch_2

num_gpus=$(( $(echo $CUDA_VISIBLE_DEVICES | tr -cd , | wc -c) + 1))

which python3
python3 spatialtrain.py --config_list binaural_train_manifests/v10_control_arch_manifest.pkl \
                 --job_id $SLURM_ARRAY_TASK_ID \
                 --gpus $num_gpus --n_jobs $SLURM_CPUS_PER_TASK --resume_training  True \
                 --exp_dir attn_cue_models \
