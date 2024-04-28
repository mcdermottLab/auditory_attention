#!/bin/bash -l
#SBATCH --job-name=make_unfamiliar_stim
#SBATCH --output=outLogs/make_swc_unfamiliar_distractor_stim_2024_%A_%a.out
#SBATCH --error=outLogs/make_swc_unfamiliar_distractor_stim_2024_%A_%a.err
#SBATCH --mem=3Gb 
#SBATCH --time=00:10:00
#SBATCH --partition=use-everything
#SBATCH --cpus-per-task=1
#SBATCH --array=15# 0-15

module load openmind8/anaconda/3-2022.10

source activate /om2/user/imgriff/conda_envs/torch_11_cuda_11_pitch

python src/get_swc_unfamiliar_distractor_stim_2024.py --array_ix $SLURM_ARRAY_TASK_ID
