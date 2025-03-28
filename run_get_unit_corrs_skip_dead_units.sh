#!/bin/bash -l
#SBATCH --job-name=get_corrs_skip_dead_units
#SBATCH --output=outLogs/get_corrs_skip_dead_units_%A_%a.out
#SBATCH --error=outLogs/get_corrs_skip_dead_units_%A_%a.err
#SBATCH --mem=8Gb                           
#SBATCH --cpus-per-task=2
#SBATCH --time=1:00:00
#SBATCH --partition=mcdermott
#SBATCH --array 13, #0-13
module load openmind8/anaconda/3-2022.10
export HDF5_USE_FILE_LOCKING=FALSE

source activate /om2/user/imgriff/conda_envs/pytorch_2_sva

python3 get_corrs_skip_dead_units.py --model '' --job_ix $SLURM_ARRAY_TASK_ID --time_avg

