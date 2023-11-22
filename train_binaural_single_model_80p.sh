#!/bin/bash
#SBATCH --job-name=co_located_80p
#SBATCH --output=outLogs/train_binaural_attn_co_locate_80p_%j.out
#SBATCH --error=outLogs/train_binaural_attn_co_locate_80p_%j.err
#SBATCH --mem=500Gb
#SBATCH -N 1
#SBATCH --cpus-per-task=80
#SBATCH --time=2-00:00:00
#SBATCH --partition=normal
#SBATCH --gres=gpu:tesla-v100:8
##SBATCH -w apollo001

#source /etc/profile.d/modules.sh
#module use /cm/shared/modulefiles

module load /openmind/miniconda

export HDF5_USE_FILE_LOCKING=FALSE

source activate /om2/user/imgriff/conda_envs/torch_11_cuda_11_pitch

#module add openmind/cudnn/11.5-v8.3.3.40
#module add openmind/cuda/12.3

which python3
python3 spatialtrain.py --config config/binaural_attn/word_task_mixed_cue_v04_80p_co_located.yml \
                 --gpus 8 --n_jobs 10 --resume_training True --clean_percentage 0.1\
                 --exp_dir attn_cue_models \

