#!/bin/bash
#SBATCH --job-name=jupyter_notebook
#SBATCH --output=outLogs/make_swc_mono_expt_stim_2023_%A_%a.out
#SBATCH --error=outLogs/make_swc_mono_expt_stim_2023_%A_%a.err
#SBATCH --mem=2Gb 
#SBATCH --time=1:00:00
#SBATCH --partition=normal
#SBATCH --cpus-per-task=1
#SBATCH --array=15-19 # 0-39


source activate /om2/user/imgriff/conda_envs/torch_11_cuda_11_pitch

python src/get_swc_mono_expmt_stim_2023.py --array_id $SLURM_ARRAY_TASK_ID
