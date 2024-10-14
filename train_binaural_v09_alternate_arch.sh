#!/bin/bash
#SBATCH --job-name=train_binaural_attn
#SBATCH --output=outLogs/train_binaural_attn_v09_alt_arch_%A_%a.out
#SBATCH --error=outLogs/train_binaural_attn_v09_alt_arch_%A_%a.err
#SBATCH --mem=100GB
#SBATCH -N 1
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --partition=ou_bcs_low
#SBATCH --gres=gpu:4
#SBATCH --array=0-9# 10 models in manifest

#source /etc/profile.d/modules.sh
#module use /cm/shared/modulefiles

module load anaconda3
source activate /orcd/data/jhm/001/imgriff/conda_envs/pytorch_2


export HDF5_USE_FILE_LOCKING=FALSE

python3 spatialtrain.py --config_list binaural_train_manifests/v09_alt_arch_search_manifest.pkl --job_id $SLURM_ARRAY_TASK_ID \
                 --gpus 4 --n_jobs 16  --resume_training  True \
                 --exp_dir attn_cue_models \
