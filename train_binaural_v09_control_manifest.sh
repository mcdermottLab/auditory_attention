#!/bin/bash
#SBATCH --job-name=train_binaural_attn
#SBATCH --output=outLogs/train_binaural_attn_v09_control_%A_%a.out
#SBATCH --error=outLogs/train_binaural_attn_v09_control_%A_%a.err
#SBATCH --mem=100GB
#SBATCH -N 1
#SBATCH --cpus-per-task=16
#SBATCH --time=2-00:00:00
#SBATCH --partition=normal
#SBATCH --gres=gpu:a100:4
#SBATCH --array=0,1 # 0-3; 3 models in manifest

#source /etc/profile.d/modules.sh
#module use /cm/shared/modulefiles

source /etc/profile.d/modules.sh
module load openmind8/anaconda/3-2022.10

export HDF5_USE_FILE_LOCKING=FALSE

source activate /om2/user/imgriff/conda_envs/pytorch_2

#module add openmind/cudnn/11.5-v8.3.3.40
#module add openmind/cuda/12.3
# 

which python3
python3 spatialtrain.py --config_list /om2/user/imgriff/projects/torch_2_aud_attn/binaural_train_manifests/v09_control_arch_manifest.pkl --job_id $SLURM_ARRAY_TASK_ID \
                 --gpus 4 --n_jobs 16  --resume_training  True \
                 --exp_dir /om2/user/imgriff/projects/torch_2_aud_attn/attn_cue_models \
