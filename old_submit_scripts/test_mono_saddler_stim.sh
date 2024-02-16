#!/bin/bash
#SBATCH --job-name=eval_mono_saddler
#SBATCH --output=outLogs/mono_saddler_test_stim_%A_%a.out
#SBATCH --error=outLogs/mono_saddler_test_stim_%A_%a.err
#SBATCH --mem=8Gb
#SBATCH --cpus-per-task=4
#SBATCH --partition=normal
#SBATCH --gres=gpu:1 --constraint=20GB
#SBATCH --array=0-36

source activate /om2/user/imgriff/conda_envs/torch_11_cuda_11_pitch

python3 eval_saddler_word_rec.py --config config/attentional_cue/attn_cue_lr_1e-4_bs_64_constrained_slope_multi_distractor.yaml \
                 --ckpt_path attn_cue_models/attn_cue_jsin_multi_distractor_w_audioset_bs_64_lr_1e-4/checkpoints/epoch=0-step=70000.ckpt \
                 --array_id $SLURM_ARRAY_TASK_ID \
                 --n_jobs 4 --exp_dir saddler_eval/ \

