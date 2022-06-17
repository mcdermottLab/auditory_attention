#!/bin/bash
#SBATCH --job-name=word_rec
#SBATCH --output=outLogs/cochlear_word_rec%j.out
#SBATCH --error=outLogs/cochlear_word_rec%j.err
#SBATCH --mem=315Gb
#SBATCH -N 1
#SBATCH --cpus-per-task=32
#SBATCH --time=120:00:00
#SBATCH --partition=normal
#SBATCH --gres=gpu:QUADRORTX6000:4

module add openmind/miniconda
module add openmind/cudnn/9.1-7.0.5
module add openmind/cuda/9.1

source activate /om2/user/imgriff/conda_envs/torchaudio_11

python3 train.py --config config/coch_word_rec/word_rec_rep_on_gpu_bs_64.yaml --gpus 4 --n_jobs 8 --mixed_precision \
                 --exp_dir ./word_rec/jsin_precombined_gammatone_40_channels_20kHz_on_gpu_1e-4lr_6_7_2022 \
                 
