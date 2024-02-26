#!/bin/bash -l
#SBATCH --job-name=get_null_corrs
#SBATCH --output=outLogs/get_fg_bg_act_%A_%a.out
#SBATCH --error=outLogs/get_fg_bg_acts_%A_%a.err
#SBATCH --mem=30Gb                           
#SBATCH --cpus-per-task=8
#SBATCH --time=2:00:00
#SBATCH --partition=normal
#SBATCH --array=1 # 0-8 for full
#SBATCH -x dgx001,dgx002,node043

module load openmind8/anaconda/3-2022.10
export HDF5_USE_FILE_LOCKING=FALSE

source activate /om2/user/imgriff/conda_envs/pytorch_2

which python3
python3 get_stage_of_attn_null_corrs.py --h5_path binaural_model_attn_stage_reps/word_task_half_co_loc_v07/word_task_half_co_loc_v07_model_activations_0dB.h5 \
                 --job_id $SLURM_ARRAY_TASK_ID \
                 --n_boot 10000 \
                 --n_jobs 8 

