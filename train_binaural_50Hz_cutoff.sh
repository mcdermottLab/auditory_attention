#!/bin/bash
#SBATCH --job-name=train_new_binaural_attn
#SBATCH --output=outLogs/word_task_half_co_loc_v09_50Hz_cutoff_%j.out
#SBATCH --error=outLogs/word_task_half_co_loc_v09_50Hz_cutoff_%j.err 
#SBATCH --mem=100Gb
#SBATCH -N 1
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=16
#SBATCH --partition=mcdermott   # multi-gpu
#SBATCH --gres=gpu:a100:4

#source /etc/profile.d/modules.sh
#module use /cm/shared/modulefiles

source /etc/profile.d/modules.sh
module load openmind8/anaconda/3-2022.10

export HDF5_USE_FILE_LOCKING=FALSE

source activate /om2/user/imgriff/conda_envs/pytorch_2
#module add openmind/cudnn/11.5-v8.3.3.40
#module add openmind/cuda/12.3

which python3

python3 spatialtrain.py --config config/binaural_attn/word_task_half_co_loc_v09_50Hz_cutoff.yaml \
                 --gpus 4 --n_jobs $SLURM_CPUS_PER_TASK --resume_training True \
                 --exp_dir attn_cue_models \

