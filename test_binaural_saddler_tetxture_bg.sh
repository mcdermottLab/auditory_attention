#!/bin/bash -l 
#SBATCH --job-name=eval_binaural_textures
#SBATCH --output=outLogs/mono_texture_test_stim_%A_%a.out
#SBATCH --error=outLogs/mono_texture_test_stim_%A_%a.err
#SBATCH --mem=12Gb
#SBATCH --cpus-per-task=4
#SBATCH --time=3:00:00
#SBATCH --partition=mcdermott
#SBATCH --gres=gpu:1 --constraint=20GB
#SBATCH --array=0 # 0-2
#SBATCH -x dgx001,dgx002

module load openmind8/anaconda/3-2022.10
export HDF5_USE_FILE_LOCKING=FALSE

source activate /om2/user/imgriff/conda_envs/pytorch_2

#                  --n_jobs 4 --exp_dir texture_mono_eval/ \

python3 eval_texture_backgrounds.py --config config/binaural_attn/word_task_v10_main_feature_gain_config.yaml \
                 --ckpt_path attn_cue_models/word_task_v10_main_feature_gain_config/checkpoints/epoch=1-step=24679.ckpt \
                 --array_id $SLURM_ARRAY_TASK_ID \
                 --n_jobs 4 --exp_dir texture_mono_eval/ \



