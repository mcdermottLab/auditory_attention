#!/bin/bash -l 
#SBATCH --job-name=eval_binaural_swc
#SBATCH --output=outLogs/binaural_swc_test_stim_%A_%a.out
#SBATCH --error=outLogs/binaural_swc_test_stim_%A_%a.err
#SBATCH --mem=12Gb
#SBATCH --cpus-per-task=4
#SBATCH --time=0:20:00
#SBATCH --partition=use-everything
#SBATCH --gres=gpu:1 --constraint=20GB
#SBATCH --array=0-40 # 0-40 # 0-40
#SBATCH -x dgx001,dgx002


module load openmind8/anaconda/3-2022.10
export HDF5_USE_FILE_LOCKING=FALSE

source activate /om2/user/imgriff/conda_envs/pytorch_2

# python3 eval_swc_mono_stim.py --config config/attentional_cue/attn_cue_lr_1e-4_bs_64_constrained_slope_multi_distractor.yaml \
#                  --ckpt_path attn_cue_models/attn_cue_jsin_multi_distractor_w_audioset_bs_64_lr_1e-4/checkpoints/epoch=0-step=70000.ckpt \
#                  --array_id $SLURM_ARRAY_TASK_ID \
#                  --n_jobs 4 --exp_dir swc_mono_eval/ \

python3 eval_swc_mono_stim.py --config config/binaural_attn/word_task_standard_v08.yaml \
                 --ckpt_path attn_cue_models/word_task_standard_v08/checkpoints/epoch=3-step=53756.ckpt \
                 --array_id $SLURM_ARRAY_TASK_ID \
                 --n_jobs 4 --exp_dir swc_mono_eval/ \

