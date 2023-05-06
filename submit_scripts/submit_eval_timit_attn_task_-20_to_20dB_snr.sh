#!/bin/bash
#SBATCH --job-name=multi_timit
#SBATCH --output=outLogs/attn_eval_timit_%j.out
#SBATCH --error=outLogs/attn_eval_timit_%j.err
#SBATCH --mem=20Gb
#SBATCH --cpus-per-task=5
#SBATCH --time=3:00:00
#SBATCH --partition=normal
#SBATCH --gres=gpu:1

module add openmind/miniconda
module add openmind/cudnn/11.5-v8.3.3.40 
module add openmind/cuda/11.3

source activate /om2/user/imgriff/conda_envs/torch_11_cuda_11_pitch

python3 eval_timit.py --gpus 1 --n_jobs 5 --exp_dir "attn_cue_models/attn_timit_task" \
                      --model_name "MultiDistractorAttnCNN_-20_to_20dB" \
                      --config_name "config/attentional_cue/attn_cue_lr_1e-4_bs_64_constrained_slope_multi_distractor_-20_to_20_SNR.yaml" \
                      --ckpt_path "attn_cue_models/attn_cue_jsin_multi_distractor_w_audioset_full_SNR_-20_to_20_bs_64_lr_1e-4/checkpoints/epoch=1-step=145791.ckpt" \
                      --whisper

