#!/bin/bash -l
#SBATCH --job-name=unit_anova
#SBATCH --output=outLogs/unit_anova_%A_%a.out
#SBATCH --error=outLogs/unit_anova_%A_%a.err
#SBATCH --mem=12Gb                           
#SBATCH --cpus-per-task=12
#SBATCH --time=4:00:00
#SBATCH --partition=mcdermott
#SBATCH --array=0-6 # 0-6 for full
#SBATCH -x dgx001,dgx002,node043,node091,node093

module load openmind8/anaconda/3-2022.10
export HDF5_USE_FILE_LOCKING=FALSE

source activate /om2/user/imgriff/conda_envs/pytorch_2_sva


python3 src/unit_tuning_anova.py --model_name "word_task_v10_main_feature_gain_config_latest_ckpt" \
                --analysis_dir "binaural_unit_activation_analysis" \
                --layer_ix $SLURM_ARRAY_TASK_ID \
                --n_jobs 12 \

# python3 src/unit_tuning_anova.py --model_name "word_task_v09_cue_loc_task" \
#                 --analysis_dir "binaural_unit_activations" \
#                 --layer_ix $SLURM_ARRAY_TASK_ID \
#                 --n_jobs 12 \

# python3 src/unit_tuning_anova.py --model_name "word_task_v09_control_no_attn" \
#                 --analysis_dir "binaural_unit_activations" \
#                 --layer_ix $SLURM_ARRAY_TASK_ID \
#                 --n_jobs 12 \

# python3 src/unit_tuning_anova.py --model_name "word_task_conventional_layer_order" \
#                 --analysis_dir "binaural_unit_activations" \
#                 --layer_ix $SLURM_ARRAY_TASK_ID \
#                 --n_jobs 12 \