#!/bin/bash
#SBATCH --job-name=multi_distractor_w_audioset
#SBATCH --output=outLogs/attn_cue_multi_distractor_w_audioset_%j.out
#SBATCH --error=outLogs/attn_cue_multi_distractor_w_audioset_%j.err
#SBATCH --mem=300Gb
#SBATCH -N 1
##SBATCH -w dgx002
##SBATCH -w node[100-115]
#SBATCH --cpus-per-task=20
#SBATCH --time=94:00:00
#SBATCH --partition=mcdermott
#SBATCH --gres=gpu:1 --constraint=30GB 

module add openmind/miniconda
module add openmind/cudnn/9.1-7.0.5
module add openmind/cuda/11.3

source activate /om2/user/imgriff/conda_envs/torch_11_cuda_11_pitch

python3 train.py --config config/attentional_cue/attn_cue_lr_1e-4_bs_64_constrained_slope_multi_distractor_-20_to_20_SNR.yaml\
                 --gpus 2 --n_jobs 10 --mixed_precision  \
                 --exp_dir ./attn_cue_models/attn_cue_jsin_multi_distractor_w_audioset_full_SNR_-20_to_20_bs_64_lr_1e-4 \
                
