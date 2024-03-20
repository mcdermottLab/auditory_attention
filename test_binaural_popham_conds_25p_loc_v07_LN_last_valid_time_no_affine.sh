#!/bin/bash -l 
#SBATCH --job-name=eval_popham
#SBATCH --output=outLogs/binaural_popham_conds_%A_%a.out
#SBATCH --error=outLogs/binaural_popham_conds_%A_%a.err
#SBATCH --mem=24Gb
#SBATCH --cpus-per-task=2
#SBATCH --time=3:00:00
#SBATCH --partition=use-everything
#SBATCH --gres=gpu:1 --constraint=12GB
#SBATCH --array=0-5 # 0-5 total


module load openmind8/anaconda/3-2022.10
export HDF5_USE_FILE_LOCKING=FALSE

source activate /om2/user/imgriff/conda_envs/pytorch_2


python3 eval_timit.py --gpus 1 --n_jobs 2 --exp_dir popham_mono_eval \
                      --test_manifest "/om2/user/imgriff/projects/torch_2_aud_attn/timit_popham_2018_test_conditions.pkl" \
                      --model_name "word_task_25p_loc_v07_LN_last_valid_time_no_affine" \
                      --config_name "/om2/user/imgriff/projects/torch_2_aud_attn/config/binaural_attn/word_task_25p_loc_v07_LN_last_valid_time_no_affine.yaml" \
                      --ckpt_path "/om2/user/imgriff/projects/torch_2_aud_attn/attn_cue_models/word_task_25p_loc_v07_LN_last_valid_time_no_affine/checkpoints/epoch=3-step=49432.ckpt" \
                      --array_id $SLURM_ARRAY_TASK_ID