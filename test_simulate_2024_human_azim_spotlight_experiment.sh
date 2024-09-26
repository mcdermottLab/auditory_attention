#!/bin/bash -l
#SBATCH --job-name=eval_model
#SBATCH --output=outLogs/sim_human_azim_spotlight_exmpt_%A_%a.out
#SBATCH --error=outLogs/sim_human_azim_spotlight_exmpt_%A_%a.err
#SBATCH --mem=8Gb
#SBATCH --cpus-per-task=2
#SBATCH --time=0:15:00
#SBATCH --partition=use-everything
#SBATCH --gres=gpu:1 --constraint=20GB
#SBATCH --array=0-26 #0-26 for all azimuth conditions 
#SBATCH -x dgx001,dgx002

module load openmind8/anaconda/3-2022.10
export HDF5_USE_FILE_LOCKING=FALSE

source activate /om2/user/imgriff/conda_envs/pytorch_2

which python3
rm -r /tmp/torchinductor_imgriff

# python3 eval_symmetric_distractors.py --config config/binaural_attn/word_task_half_co_loc_v08_gender_bal_4M_sanity.yaml \
#                  --ckpt_path attn_cue_models/word_task_half_co_loc_v08_gender_bal_4M_sanity/checkpoints/epoch=7-step=89878.ckpt \
#                  --test_manifest binaural_test_manifests/sim_human_azim_spotlight_experiment_min_reverb_mit_room.pkl \
#                  --location_idx $SLURM_ARRAY_TASK_ID \
#                  --gpus 1 --n_jobs 2 --exp_dir binaural_eval/simulate_2024_human_azimuth_spotlight_experiment \
#                  --cue_type voice_and_location --no-overwrite --n_per_job 1 --sim_human_array_exmpt --run_all_stim 

python3 eval_symmetric_distractors.py --config config/binaural_attn/word_task_half_co_loc_v08_gender_bal_4M_sanity.yaml \
                 --ckpt_path attn_cue_models/word_task_half_co_loc_v08_gender_bal_4M_sanity/checkpoints/epoch=7-step=89878.ckpt \
                 --test_manifest binaural_test_manifests/sim_human_azim_spotlight_experiment_anechoic_room.pkl \
                 --location_idx $SLURM_ARRAY_TASK_ID \
                 --gpus 1 --n_jobs 2 --exp_dir binaural_eval/simulate_2024_human_azimuth_spotlight_experiment_anechoic \
                 --cue_type voice_and_location --no-overwrite --n_per_job 1 --sim_human_array_exmpt --run_all_stim 

