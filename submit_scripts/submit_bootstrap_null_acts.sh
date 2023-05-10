#!/bin/bash
#SBATCH --job-name=bootstrap
#SBATCH --output=outLogs/bootstrap_null_corrs_%A_%a.out
#SBATCH --error=outLogs/bootstrap_null_corrs_%A_%a.err
#SBATCH --mem=80Gb
#SBATCH -c 1
#SBATCH --time=1:00:00
#SBATCH --partition=normal
#SBATCH --array=1-1000%100


module add openmind/miniconda

source activate /om2/user/imgriff/conda_envs/torch_11_cuda_11


python src/make_null_activation_corr_distributions.py -N 10 -R $SLURM_ARRAY_TASK_ID
