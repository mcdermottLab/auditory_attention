#!/bin/bash
#SBATCH --job-name=finetune_localization
#SBATCH --output=outLogs/finetune_binaural_attn_v07_%A_%a.out
#SBATCH --error=outLogs/finetune_binaural_attn_v07_%A_%a.err
#SBATCH --mem=24GB
#SBATCH -N 1
#SBATCH --cpus-per-task=4
#SBATCH --time=1-00:00:00
#SBATCH --partition=use-everything
#SBATCH --gres=gpu:1 --constraint=40GB
#SBATCH --array=0-6

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
python3 finetune_for_localization.py --config config/binaural_attn/word_task_standard_v07.yaml \
                                     --ckpt_path attn_cue_models/word_task_standard_v07/checkpoints/epoch=3-step=67111.ckpt \
                                     --gpus 1 --n_jobs 4  \
                                     --exp_dir finetune_localization \
                                     --array_id $SLURM_ARRAY_TASK_ID
