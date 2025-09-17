#!/bin/bash -l
#SBATCH --job-name=get_unit_acts
#SBATCH --output=outLogs/get_unit_acts_diff_sources_%A_%a.out
#SBATCH --error=outLogs/get_unit_acts_diff_sources_%A_%a.err
#SBATCH --mem=20Gb                           
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --partition=use-everything
#SBATCH --gres=gpu:a100:1 
#SBATCH --array=12  # 0-12  # 0-12

module load openmind8/anaconda/3-2022.10
export HDF5_USE_FILE_LOCKING=FALSE

source activate /om2/user/imgriff/conda_envs/pytorch_2

which python3
rm -r /tmp/torchinductor_imgriff


rsync_file() {
    local file="$1"
    local rel_path="${file#$SRC_DIR/}"  # Relative path
    local dest_path="$DEST/$(dirname "$rel_path")"

    # Create remote directory structure and rsync the file
    echo "$file" 
}


                

python3 get_acts_for_tuning_and_selection_analysis_as_fn_of_separation_symmetric.py --config_list model_architecture_activation_manifests/all_v10_architectures_alts_and_controls.pkl \
                --n_activations 500 \
                --n_jobs 8 \
                --job_id $SLURM_ARRAY_TASK_ID \
                --cue_single_source \
                --output_dir binaural_unit_activation_analysis_symmetric_distractor \
                --resume_progress \
                --overwrite \
                # --random_weights
                # --diotic \



