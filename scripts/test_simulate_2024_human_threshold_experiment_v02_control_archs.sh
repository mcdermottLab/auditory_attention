#!/bin/bash -l
#SBATCH --job-name=eval_model
#SBATCH --output=outLogs/sim_human_threshold_exmpt_v02_control_archs_%A_%a.out
#SBATCH --error=outLogs/sim_human_threshold_exmpt_v02_control_archs_%A_%a.err
#SBATCH --mem=8Gb
#SBATCH --cpus-per-task=4
#SBATCH --time=1:15:00
#SBATCH --partition=use-everything
#SBATCH --gres=gpu:1 --constraint=20GB
#SBATCH --array=0-59 #0-59 for all azimuth conditions 
#SBATCH -x dgx001,dgx002

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"


module load openmind8/anaconda/3-2022.10
export HDF5_USE_FILE_LOCKING=FALSE

source activate /om2/user/imgriff/conda_envs/pytorch_2

which python3


rm -rf /tmp/torchinductor_imgriff
python3 -m src.eval_sim_array_threshold_experiment_v02 --config config/binaural_attn/word_task_early_only_v10.yaml \
                 --ckpt_path attn_cue_models/word_task_early_only_v10/checkpoints/epoch=7-step=92753.ckpt \
                 --test_manifest binaural_test_manifests/sim_human_threshold_experiment_v02_min_reverb_room1006_30dB_bg_noise.pkl \
                 --location_idx $SLURM_ARRAY_TASK_ID \
                 --gpus 1 --n_jobs 4 --exp_dir binaural_eval/simulate_2024_human_threshold_experiment_v02_30_dB_pink_noise_min_verb_mit46_1004 \
                 --cue_type voice_and_location --no-overwrite --n_per_job 1 --sim_human_array_exmpt --run_all_stim --pink_noise_distractor

rm -rf /tmp/torchinductor_imgriff
python3 -m src.eval_sim_array_threshold_experiment_v02 --config config/binaural_attn/word_task_late_only_v10.yaml \
                 --ckpt_path attn_cue_models/word_task_late_only_v10/checkpoints/epoch=7-step=96753.ckpt \
                 --test_manifest binaural_test_manifests/sim_human_threshold_experiment_v02_min_reverb_room1006_30dB_bg_noise.pkl \
                 --location_idx $SLURM_ARRAY_TASK_ID \
                 --gpus 1 --n_jobs 4 --exp_dir binaural_eval/simulate_2024_human_threshold_experiment_v02_30_dB_pink_noise_min_verb_mit46_1004 \
                 --cue_type voice_and_location --no-overwrite --n_per_job 1 --sim_human_array_exmpt --run_all_stim --pink_noise_distractor

rm -rf /tmp/torchinductor_imgriff
python3 -m src.eval_sim_array_threshold_experiment_v02 --config config/binaural_attn/word_task_v10_control_no_attn.yaml \
                 --ckpt_path attn_cue_models/word_task_v10_control_no_attn/checkpoints/epoch=7-step=94753.ckpt \
                 --test_manifest binaural_test_manifests/sim_human_threshold_experiment_v02_min_reverb_room1006_30dB_bg_noise.pkl \
                 --location_idx $SLURM_ARRAY_TASK_ID \
                 --gpus 1 --n_jobs 4 --exp_dir binaural_eval/simulate_2024_human_threshold_experiment_v02_30_dB_pink_noise_min_verb_mit46_1004 \
                 --cue_type voice_and_location --no-overwrite --n_per_job 1 --sim_human_array_exmpt --run_all_stim --pink_noise_distractor
