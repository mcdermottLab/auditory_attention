#!/bin/bash
#SBATCH --job-name=train_binaural_attn
#SBATCH --output=outLogs/train_binaural_attn_v05_%A_%a.out
#SBATCH --error=outLogs/train_binaural_attn_v05_%A_%a.err
#SBATCH --mem=100GB
#SBATCH -N 1
#SBATCH --cpus-per-task=16
#SBATCH --time=2-00:00:00
#SBATCH --partition=normal
#SBATCH --gres=gpu:a100:4
#SBATCH --array=0#-2 # 0-2; 3 models in manifest

#source /etc/profile.d/modules.sh
#module use /cm/shared/modulefiles

source /etc/profile.d/modules.sh
module load openmind8/anaconda/3-2022.10

export HDF5_USE_FILE_LOCKING=FALSE

source activate /om2/user/imgriff/conda_envs/pytorch_2

#module add openmind/cudnn/11.5-v8.3.3.40
#module add openmind/cuda/12.3

which python3
python3 spatialtrain.py --config_list binaural_train_manifests/v05_models_12_2023.pkl --job_id $SLURM_ARRAY_TASK_ID\
                 --gpus 4 --n_jobs 4 --resume_training True \
                 --exp_dir attn_cue_models \
