#!/bin/bash -l
#SBATCH --job-name=eval_model
#SBATCH --output=outLogs/sim_byrne_et_al_2023_%A_%a.out
#SBATCH --error=outLogs/sim_byrne_et_al_2023_%A_%a.err
#SBATCH --mem=8Gb
#SBATCH --cpus-per-task=4
#SBATCH --time=3:00:00
#SBATCH --partition=use-everything
#SBATCH --gres=gpu:1 --constraint=20GB
#SBATCH --array=0-44 #0-44 for all azimuth conditions 
#SBATCH -x dgx001,dgx002

module load openmind8/anaconda/3-2022.10
export HDF5_USE_FILE_LOCKING=FALSE

source activate /om2/user/imgriff/conda_envs/pytorch_2

which python3
rm -r /tmp/torchinductor_imgriff

python3 eval_symmetric_distractors.py --config config/binaural_attn/word_task_v10_gain_post_norm_config.yaml \
                 --ckpt_path attn_cue_models/word_task_v10_gain_post_norm_config/checkpoints/epoch=0-step=2000-v5.ckpt \
                 --test_manifest binaural_test_manifests/simulate_byrne_et_al_2023.pkl \
                 --location_idx $SLURM_ARRAY_TASK_ID \
                 --gpus 1 --n_jobs 4 --exp_dir binaural_eval/simulate_byrne_et_al_2023 \
                 --cue_type voice_and_location --no-overwrite --n_per_job 1 --sim_human_array_exmpt --run_all_stim
