#!/bin/bash
#SBATCH --job-name=forced_align
#SBATCH --output=outLogs/forced_alignment_%A_%a.out
#SBATCH --error=outLogs/forced_alignment_%A_%a.err
#SBATCH --mem=6Gb
#SBATCH --cpus-per-task=1
#SBATCH --time=0:40:00
#SBATCH --partition=normal
#SBATCH --gres=gpu:1
#SBATCH --array=0-100


module add openmind/miniconda
module add openmind/cudnn/11.5-v8.3.3.40
module add openmind/cuda/11.3


export CONDA_ENVS_PATH=~/my-envs:/om2/user/imgriff/conda_envs

source activate /om2/user/imgriff/conda_envs/torch_11_cuda_11_pitch

python util/wav2vec2_forced_alignment.py  --array_ix $SLURM_ARRAY_TASK_ID \
                                          --batch_size 15500 \
