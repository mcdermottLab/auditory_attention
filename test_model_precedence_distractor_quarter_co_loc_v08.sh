#!/bin/bash -l
#SBATCH --job-name=eval_model
#SBATCH --output=outLogs/eval_precedence_quarter_co_loc_v08_%A_%a.out
#SBATCH --error=outLogs/eval_precedence_quarter_co_loc_v08_%A_%a.err
#SBATCH --mem=10Gb
#SBATCH --cpus-per-task=2
#SBATCH --time=1:00:00
#SBATCH --partition=use-everything
#SBATCH --gres=gpu:1 --constraint=20GB
#SBATCH --array=0-29 # 0-29 if running 5 per job; 0-8 for just freyman conds
#SBATCH -x dgx001,dgx002

module load openmind8/anaconda/3-2022.10
export HDF5_USE_FILE_LOCKING=FALSE

source activate /om2/user/imgriff/conda_envs/pytorch_2

which python3
python3 eval_precedence.py --config /om2/user/imgriff/projects/torch_2_aud_attn/config/binaural_attn/word_task_quarter_co_loc_v08.yaml \
                 --ckpt_path /om2/user/imgriff/projects/torch_2_aud_attn/attn_cue_models/word_task_quarter_co_loc_v08/checkpoints/epoch=1-step=21252.ckpt \
                 --test_manifest /om2/user/imgriff/projects/torch_2_aud_attn/binaural_test_manifests/freymen_1999_test_conds.pkl \
                 --model_name word_task_quarter_co_loc_v08 --location_idx $SLURM_ARRAY_TASK_ID \
                 --gpus 1 --n_jobs 2 --exp_dir binaural_eval/precedence_distractor_test \
                 --cue_type voice_and_location --overwrite --n_per_job 5



# which python3
# python3 eval_precedence.py --config config/binaural_attn/word_task_standard_v07.yaml \
#                  --ckpt_path attn_cue_models/word_task_standard_v07/checkpoints/epoch=3-step=65111.ckpt \
#                  --test_manifest binaural_test_manifests/precedence_distractor_conditions.pkl \
#                  --model_name word_task_standard_v07 --location_idx $SLURM_ARRAY_TASK_ID \
#                  --gpus 1 --n_jobs 2 --exp_dir binaural_eval/precedence_distractor_test \
#                  --cue_type voice_and_location --overwrite --n_per_job 5