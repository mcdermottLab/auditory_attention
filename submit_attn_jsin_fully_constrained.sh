#!/bin/bash
#SBATCH --job-name=attn_cue
#SBATCH --output=outLogs/attn_cue_audioset_%j.out
#SBATCH --error=outLogs/attn_cue_audioset_%j.err
#SBATCH --mem=200Gb
#SBATCH -N 1
#SBATCH --cpus-per-task=40
#SBATCH --time=72:00:00
#SBATCH --partition=mcdermott
#SBATCH --gres=gpu:tesla-v100:4
#SBATCH -w dgx002

module add openmind/miniconda
module add openmind/cudnn/11.5-v8.3.3.40
module add openmind/cuda/11.3

source activate /om2/user/imgriff/conda_envs/torch_11_cuda_11

python3 train.py --config config/attentional_cue/attn_cue_lr_1e-4_bs_64_constrained_slope_noise_only.yaml \
                 --gpus 4 --n_jobs 10 --mixed_precision --dgx002_path \
                 --exp_dir ./attn_cue_models/attn_cue_jsin_audset_bg_fully_constrained_bs_64_lr_1e-4 \
                 
