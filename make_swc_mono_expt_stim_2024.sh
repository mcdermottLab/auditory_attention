#!/bin/bash -l
#SBATCH --job-name=make_swc_stim
#SBATCH --output=outLogs/make_swc_mono_expt_stim_2024_%A_%a.out
#SBATCH --error=outLogs/make_swc_mono_expt_stim_202_%A_%a.err
#SBATCH --mem=4Gb 
#SBATCH --time=1:00:00
#SBATCH --partition=use-everything
#SBATCH --cpus-per-task=1
#SBATCH --array=60# 0-60

module load openmind8/anaconda/3-2022.10
source activate /om2/user/imgriff/conda_envs/torch_11_cuda_11_pitch

python3 src/get_swc_mono_expmt_stim_2024.py --array_id $SLURM_ARRAY_TASK_ID
