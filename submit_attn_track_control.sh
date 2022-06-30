#!/bin/bash
#SBATCH --job-name=word_rec
#SBATCH --output=outLogs/cochlear_attn_tracking_control%j.out
#SBATCH --error=outLogs/cochlear_attn_tracking_control%j.err
#SBATCH --mem=315Gb
#SBATCH -N 1
#SBATCH --cpus-per-task=32
#SBATCH --time=72:00:00
#SBATCH --partition=mcdermott
#SBATCH --gres=gpu:QUADRORTX6000:4

module add openmind/miniconda
module add openmind/cudnn/9.1-7.0.5
module add openmind/cuda/9.1

source activate /om2/user/imgriff/conda_envs/torchaudio_11

python3 train.py --config config/attentional_cue/attn_tracking_control.yaml --gpus 4 --n_jobs 8 --mixed_precision \
                 --exp_dir ./multi_talker_control/jsin_precombined_gammatone_40_channels_20kHz_on_gpu_1e-4lr \
                 
