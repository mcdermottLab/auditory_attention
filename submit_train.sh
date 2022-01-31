#!/bin/bash
#SBATCH --job-name=LAS
#SBATCH --output=outLogs/LAS_librispeech_%j.out
#SBATCH --error=outLogs/LAS_librispeech_%j.err
#SBATCH --mem=20Gb
#SBATCH --cpus-per-task=4
#SBATCH --time=94:00:00
#SBATCH --partition=mcdermott
#SBATCH --gres=gpu:GEFORCEGTX1080TI:1


module add openmind/miniconda/2020-01-29-py3.7
module add openmind/cudnn/9.1-7.0.5
module add openmind/cuda/9.1


export CONDA_ENVS_PATH=~/my-envs:/om4/group/mcdermott/user/imgriff/conda_envs_files

source activate /om4/group/mcdermott/user/imgriff/conda_envs_files/pytorch_ASR



python3 main.py --config config/libri/lib_word_example.yaml --njobs 4

