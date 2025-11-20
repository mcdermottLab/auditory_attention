#!/bin/bash -l
#SBATCH --job-name=eval_model
#SBATCH --output=outLogs/sim_human_azim_spotlight_exmpt_alt_arch_%A_%a.out
#SBATCH --error=outLogs/sim_human_azim_spotlight_exmpt_alt_arch_%A_%a.err
#SBATCH --mem=12Gb
#SBATCH --cpus-per-task=2
#SBATCH --time=2:00:00
#SBATCH --partition=use-everything
#SBATCH --gres=gpu:1 --constraint=20GB
#SBATCH --array=0-15 #0-15 for all azimuth conditions 
#SBATCH -x dgx001,dgx002

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"


module load openmind8/anaconda/3-2022.10
export HDF5_USE_FILE_LOCKING=FALSE

source activate /om2/user/imgriff/conda_envs/pytorch_2

which python3
rm -r /tmp/torchinductor_imgriff

python3 -m src.eval_sim_array_spotlight_experiment_v02 --config config/binaural_attn/word_task_v10_main_feature_gain_config.yaml \
                 --ckpt_path attn_cue_models/word_task_v10_main_feature_gain_config/checkpoints/epoch=1-step=24679-v1.ckpt \
                 --test_manifest binaural_test_manifests/sim_azim_spotlight_v02.pkl \
                 --location_idx $SLURM_ARRAY_TASK_ID \
                 --gpus 1 --n_jobs 2 --exp_dir binaural_eval/sim_azim_spotlight_v02_min_reverb_room1004_30dB_pink_noise_bg_ssn \
                 --cue_type voice_and_location --no-overwrite --n_per_job 1 --run_all_stim --pink_noise_bg --ssn_distractor
