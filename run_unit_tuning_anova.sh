#!/bin/bash -l
#SBATCH --job-name=unit_anova
#SBATCH --output=outLogs/unit_anova_%A_%a.out
#SBATCH --error=outLogs/unit_anova_%A_%a.err
#SBATCH --mem=12Gb                           
#SBATCH --cpus-per-task=10
#SBATCH --time=2:00:00
#SBATCH --partition=use-everything
#SBATCH --array=1-6 # 0-6 for full
#SBATCH -x dgx001,dgx002,node043,node091,node093

module load openmind8/anaconda/3-2022.10
export HDF5_USE_FILE_LOCKING=FALSE

source activate /om2/user/imgriff/conda_envs/pytorch_2_sva

python3 src/unit_tuning_anova.py --model_name "word_task_half_co_loc_v08_gender_bal_4M_w_no_cue_learned_higher_lr_less_dropout" \
                --analysis_dir "binaural_unit_activations" \
                --layer_ix $SLURM_ARRAY_TASK_ID \
                --n_jobs 10 \