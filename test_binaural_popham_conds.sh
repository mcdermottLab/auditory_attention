#!/bin/bash -l 
#SBATCH --job-name=eval_popham
#SBATCH --output=outLogs/binaural_popham_conds_%A_%a.out
#SBATCH --error=outLogs/binaural_popham_conds_%A_%a.err
#SBATCH --mem=24Gb
#SBATCH --cpus-per-task=2
#SBATCH --time=3:00:00
#SBATCH --partition=use-everything
#SBATCH --gres=gpu:1 --constraint=12GB
#SBATCH --array=0-5# 0-5 total


module load openmind8/anaconda/3-2022.10
export HDF5_USE_FILE_LOCKING=FALSE

source activate /om2/user/imgriff/conda_envs/pytorch_2


python3 eval_timit.py --gpus 1 --n_jobs 2 --exp_dir popham_mono_eval \
                      --test_manifest "timit_popham_2018_test_conditions.pkl" \
                      --model_name "word_task_half_co_loc_v07" \
                      --config_name "config/binaural_attn/word_task_half_co_loc_v07.yaml" \
                      --ckpt_path "attn_cue_models/word_task_half_co_loc_v07/checkpoints/epoch=2-step=46074.ckpt" \
                      --array_id $SLURM_ARRAY_TASK_ID