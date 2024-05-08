#!/bin/bash
#SBATCH --job-name=train_new_binaural_attn
#SBATCH --output=outLogs/train_v08_gender_bal_4M_orig_%j.out
#SBATCH --error=outLogs/train_v08_gender_bal_4M_orig_%j.err
#SBATCH --mem=100Gb
#SBATCH -N 1
#SBATCH --cpus-per-task=16
#SBATCH --time=2-00:00:00
#SBATCH --partition=normal
#SBATCH --gres=gpu:a100:4
##SBATCH -w apollo001

#source /etc/profile.d/modules.sh
#module use /cm/shared/modulefiles

source /etc/profile.d/modules.sh
module load openmind8/anaconda/3-2022.10

export HDF5_USE_FILE_LOCKING=FALSE

source activate /om2/user/imgriff/conda_envs/pytorch_2
#module add openmind/cudnn/11.5-v8.3.3.40
#module add openmind/cuda/12.3

which python3
# python3 spatialtrain.py --config config/binaural_attn/word_task_half_co_locate_deep_fc_1024_v08.yaml \
#                  --gpus 4 --n_jobs 4 --resume_training True \
#                  --exp_dir attn_cue_models \

# python3 spatialtrain.py --config config/binaural_attn/word_task_half_co_loc_v08_gender_bal_4M.yaml \
#                  --gpus 4 --n_jobs 4 --resume_training True \
#                  --exp_dir attn_cue_models \

# python3 spatialtrain.py --config config/binaural_attn/word_task_half_co_loc_v08_gender_bal_4M_orig.yaml \
#                  --gpus 4 --n_jobs 4 --resume_training True \
#                  --exp_dir attn_cue_models \

python3 spatialtrain.py --config config/binaural_attn/word_task_half_co_loc_v08.yaml \
                 --gpus 4 --n_jobs 4 --resume_training True \
                 --exp_dir attn_cue_models \

