#!/bin/bash -l 
#SBATCH --job-name=eval_binaural_swc
#SBATCH --output=outLogs/binaural_swc_test_stim_%A_%a.out
#SBATCH --error=outLogs/binaural_swc_test_stim_%A_%a.err
#SBATCH --mem=12Gb
#SBATCH --cpus-per-task=4
#SBATCH --time=0:20:00
#SBATCH --partition=use-everything
#SBATCH --gres=gpu:1 --constraint=20GB
#SBATCH --array=0-40 # 0-40
#SBATCH -x dgx001,dgx002


module load openmind8/anaconda/3-2022.10
export HDF5_USE_FILE_LOCKING=FALSE

source activate /om2/user/imgriff/conda_envs/pytorch_2

# python3 eval_swc_mono_stim.py --config config/binaural_attn/word_task_25p_loc_v07_LN_last_valid_time_no_affine.yaml \
#                  --ckpt_pat attn_cue_models/word_task_25p_loc_v07_LN_last_valid_time_no_affine/checkpoints/epoch=3-step=49432.ckpt \
#                  --array_id $SLURM_ARRAY_TASK_ID \
#                  --n_jobs 4 --exp_dir swc_mono_eval/ \


python3 eval_swc_mono_stim.py --config config/binaural_attn/word_task_quarter_co_loc_v08.yaml \
                 --ckpt_pat attn_cue_models/word_task_quarter_co_loc_v08/checkpoints/epoch=1-step=25252.ckpt \
                 --array_id $SLURM_ARRAY_TASK_ID \
                 --n_jobs 4 --exp_dir swc_mono_eval/ \

