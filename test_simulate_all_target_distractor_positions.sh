#!/bin/bash -l
#SBATCH --job-name=eval_model
#SBATCH --output=outLogs/sim_all_target_dist_positions_%A_%a.out
#SBATCH --error=outLogs/sim_all_target_dist_positions_%A_%a.err
#SBATCH --mem=8Gb
#SBATCH --cpus-per-task=2
#SBATCH --time=3:00:00
#SBATCH --partition=use-everything
#SBATCH --gres=gpu:a100:1 --constraint=20GB
#SBATCH --array=1059 #0-1814 #0-1815 for 35 locations per job array ix 
#SBATCH -x dgx001,dgx002

module load openmind8/anaconda/3-2022.10
export HDF5_USE_FILE_LOCKING=FALSE

source activate /om2/user/imgriff/conda_envs/pytorch_2

which python3
rm -r /tmp/torchinductor_imgriff
python3 eval_symmetric_distractors.py --config config/binaural_attn/word_task_v10_main_feature_gain_config.yaml \
                 --ckpt_path attn_cue_models/word_task_v10_main_feature_gain_config/checkpoints/epoch=1-step=24679-v1.ckpt \
                 --test_manifest binaural_test_manifests/all_target_distractor_pairs.pkl \
                 --location_idx $SLURM_ARRAY_TASK_ID \
                 --gpus 1 --n_jobs 2 --exp_dir binaural_eval/simulate_all_target_distractor_location_pairs \
                 --cue_type voice_and_location --no-overwrite --n_per_job 35 --sim_human_array_exmpt --run_all_stim 

