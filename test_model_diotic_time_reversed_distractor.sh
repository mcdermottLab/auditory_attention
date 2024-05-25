#!/bin/bash -l
#SBATCH --job-name=eval_model
#SBATCH --output=outLogs/eval_time_reversed_distractor_%A_%a.out
#SBATCH --error=outLogs/eval_time_reversed_distractor_%A_%a.err
#SBATCH --mem=8Gb
#SBATCH --cpus-per-task=2
#SBATCH --time=00:15:00
#SBATCH --partition=use-everything
#SBATCH --gres=gpu:1 --constraint=16GB
#SBATCH --array=5-9 #0-9
#SBATCH -x dgx001,dgx002

module load openmind8/anaconda/3-2022.10
export HDF5_USE_FILE_LOCKING=FALSE

source activate /om2/user/imgriff/conda_envs/pytorch_2

which python3
# python3 eval_diotic_w_manifest.py --config config/binaural_attn/word_task_standard_v08.yaml \
#                  --ckpt_path attn_cue_models/word_task_standard_v08/checkpoints/epoch=3-step=51756-v1.ckpt \
#                  --test_manifest binaural_test_manifests/time_reversed_test_manifest_1_distractor.pkl \
#                  --model_name word_task_standard_v08 --job_idx $SLURM_ARRAY_TASK_ID \
#                  --gpus 1 --n_jobs 2 --exp_dir time_reversed_eval \
#                  --cue_type voice_and_location --no-overwrite

python3 eval_diotic_w_manifest.py --config config/binaural_attn/word_task_half_co_loc_v08_gender_bal_4M_orig.yaml \
                 --ckpt_path attn_cue_models/word_task_half_co_loc_v08_gender_bal_4M_orig/checkpoints/epoch=0-step=6000-v1.ckpt \
                 --test_manifest binaural_test_manifests/time_reversed_test_manifest_1_distractor.pkl \
                 --model_name word_task_half_co_loc_v08_gender_bal_4M_orig --job_idx $SLURM_ARRAY_TASK_ID \
                 --gpus 1 --n_jobs 2 --exp_dir time_reversed_eval \
                 --cue_type voice_and_location --no-overwrite

