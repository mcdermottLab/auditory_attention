#!/bin/bash -l
#SBATCH --job-name=eval_model
#SBATCH --output=outLogs/sim_core_target_dist_positions_ssn_dist_%A_%a.out
#SBATCH --error=outLogs/sim_core_target_dist_positions_ssn_dist_%A_%a.err
#SBATCH --mem=8Gb
#SBATCH --cpus-per-task=2
#SBATCH --time=3:00:00
#SBATCH --partition=use-everything
#SBATCH --gres=gpu:a100:1 --constraint=20GB
#SBATCH --array=65,66,67,124-128,132,138,236,326 #0-360
#SBATCH -x dgx001,dgx002

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"


module load openmind8/anaconda/3-2022.10
export HDF5_USE_FILE_LOCKING=FALSE

source activate /om2/user/imgriff/conda_envs/pytorch_2

which python3
rm -r /tmp/torchinductor_imgriff

python3 -m src.eval_symmetric_distractors --config config/binaural_attn/word_task_v10_main_feature_gain_config.yaml \
                 --ckpt_path attn_cue_models/word_task_v10_main_feature_gain_config/checkpoints/epoch=1-step=24679-v1.ckpt \
                 --test_manifest binaural_test_manifests/target_distractor_-90_to_90_azim_at_0_elev.pkl \
                 --location_idx $SLURM_ARRAY_TASK_ID \
                 --gpus 1 --n_jobs 2 --exp_dir binaural_eval/core_target_distractor_location_pairs_ssn_distractor \
                 --cue_type voice_and_location --no-overwrite --sim_human_array_exmpt --run_all_stim --ssn_distractors --n_per_job 1
