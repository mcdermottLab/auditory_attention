#!/bin/bash -l 
#SBATCH --job-name=eval_popham
#SBATCH --output=outLogs/binaural_popham_conds_quarter_co_loc_v08_%A_%a.out
#SBATCH --error=outLogs/binaural_popham_conds_quarter_co_loc_v08_%A_%a.err
#SBATCH --mem=8Gb
#SBATCH --cpus-per-task=2
#SBATCH --time=00:20:00
#SBATCH --partition=use-everything
#SBATCH --gres=gpu:1 --constraint=16GB
#SBATCH --array=0-5 #0-5 total
#SBATCH -x dgx001,dgx002


module load openmind8/anaconda/3-2022.10
export HDF5_USE_FILE_LOCKING=FALSE

source activate /om2/user/imgriff/conda_envs/pytorch_2


# python3 eval_timit.py --gpus 1 --n_jobs 2 --exp_dir popham_mono_eval \
#                       --test_manifest "/om2/user/imgriff/projects/torch_2_aud_attn/timit_popham_2018_test_conditions.pkl" \
#                       --model_name "word_task_quarter_co_loc_v08" \
#                       --config_name "/om2/user/imgriff/projects/torch_2_aud_attn/config/binaural_attn/word_task_quarter_co_loc_v08.yaml" \
#                       --ckpt_path "/om2/user/imgriff/projects/torch_2_aud_attn/attn_cue_models/word_task_quarter_co_loc_v08/checkpoints/epoch=1-step=21252.ckpt" \
#                       --array_id $SLURM_ARRAY_TASK_ID

python3 eval_timit.py --gpus 1 --n_jobs 2 --exp_dir popham_mono_eval \
                      --test_manifest "/om2/user/imgriff/projects/torch_2_aud_attn/timit_popham_2018_with_crossed_test_conditions.pkl" \
                      --model_name "word_task_quarter_co_loc_v08" \
                      --config_name "/om2/user/imgriff/projects/torch_2_aud_attn/config/binaural_attn/word_task_quarter_co_loc_v08.yaml" \
                      --ckpt_path "/om2/user/imgriff/projects/torch_2_aud_attn/attn_cue_models/word_task_quarter_co_loc_v08/checkpoints/epoch=1-step=21252.ckpt" \
                      --array_id $SLURM_ARRAY_TASK_ID
