#!/bin/bash
#SBATCH --job-name=train_binaural_attn
#SBATCH --output=outLogs/train_binaural_attn_%A_%a.out
#SBATCH --error=outLogs/train_binaural_attn_%A_%a.err
#SBATCH --mem=800Gb
#SBATCH -N 1
#SBATCH --cpus-per-task=80
#SBATCH --time=12:00:00
#SBATCH --partition=multi-gpu
#SBATCH --gres=gpu:a100:8 --constraint=60GB 
#SBATCH --array=0

source /etc/profile.d/modules.sh
module use /cm/shared/modulefiles
module add openmind/miniconda
export HDF5_USE_FILE_LOCKING=FALSE

source activate /om2/user/imgriff/conda_envs/torch_11_cuda_11_pitch

module add openmind/cudnn/11.5-v8.3.3.40
module add openmind/cuda/11.3


python3 spatialtrain.py --config_list large_archs.pkl --job_id $SLURM_ARRAY_TASK_ID\
                 --gpus 8 --n_jobs 10 --random_seed 0\
                 --exp_dir attn_cue_models \

