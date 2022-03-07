#!/bin/bash
#SBATCH --job-name=LAS
#SBATCH --output=outLogs/LAS_gigaspeech_ctc_%j.out
#SBATCH --error=outLogs/LAS_gigaspeech_ctc_%j.err
#SBATCH --mem=128Gb
#SBATCH --cpus-per-task=48
#SBATCH --time=94:00:00
#SBATCH --partition=mcdermott
#SBATCH --gres=gpu:GEFORCEGTX1080TI:1


module add openmind/miniconda/2020-01-29-py3.7
module add openmind/cudnn/9.1-7.0.5
module add openmind/cuda/9.1


export CONDA_ENVS_PATH=~/my-envs:/om4/group/mcdermott/user/imgriff/conda_envs_files

source activate /om4/group/mcdermott/user/imgriff/conda_envs_files/pytorch_ASR



python3 main.py --config config/giga/giga_word_ctc_train.yaml --njobs 48 

