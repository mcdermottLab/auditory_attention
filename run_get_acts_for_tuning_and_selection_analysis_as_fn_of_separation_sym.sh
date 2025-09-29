#!/bin/bash -l
#SBATCH --job-name=get_unit_acts
#SBATCH --output=outLogs/get_unit_acts_sym_distractor_%A_%a.out
#SBATCH --error=outLogs/get_unit_acts_sym_distractor_%A_%a.err
#SBATCH --mem=8Gb 
#SBATCH --time=4:00:00
#SBATCH --partition=ou_bcs_high
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --array=12  # 0-12  # 0-12

module add miniforge

export HDF5_USE_FILE_LOCKING=FALSE

mamba activate pytorch_2
                

python3 get_acts_for_tuning_and_selection_analysis_as_fn_of_separation_symmetric.py \
                --config_list model_architecture_activation_manifests/all_v10_architectures_alts_and_controls.pkl \
                --n_activations 500 \
                --n_jobs 8 \
                --job_id $SLURM_ARRAY_TASK_ID \
                --cue_single_source \
                --output_dir binaural_unit_activation_analysis_symmetric_distractor \
                --resume_progress \
                --overwrite \
                # --random_weights
                # --diotic \



