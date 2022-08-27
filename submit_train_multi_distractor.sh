#!/bin/bash
#SBATCH --job-name=multi_distractor
#SBATCH --output=outLogs/attn_cue_multi_distractor_%j.out
#SBATCH --error=outLogs/attn_cue_multi_distractor_%j.err
#SBATCH --mem=128Gb
#SBATCH -N 1
##SBATCH -w node093
#SBATCH  -x node[100-115]
#SBATCH --cpus-per-task=40
#SBATCH --time=72:00:00
#SBATCH --partition=mcdermott
#SBATCH --gres=gpu:4 --constraint=20GB

module add openmind/miniconda
module add openmind/cudnn/9.1-7.0.5
module add openmind/cuda/11.3

source activate /om2/user/imgriff/conda_envs/torch_11_cuda_11

python3 train.py --config config/attentional_cue/attn_cue_lr_1e-4_bs_64_constrained_slope_multi_distractor.yaml\
                 --gpus 4 --n_jobs 10 --mixed_precision \
                 --exp_dir ./attn_cue_models/attn_cue_jsin_multi_distractor_bs_64_lr_1e-4 \
                 
