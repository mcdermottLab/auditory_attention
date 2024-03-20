#!/bin/bash -l 
#SBATCH --job-name=eval_cue_duration
#SBATCH --output=outLogs/cue_duration_test_stim_%A_%a.out
#SBATCH --error=outLogs/cue_duration_test_stim_%A_%a.err
#SBATCH --mem=4Gb
#SBATCH --cpus-per-task=4
#SBATCH --time=0:20:00
#SBATCH --partition=normal
#SBATCH --gres=gpu:1 --constraint=20GB
#SBATCH --array=1-8 # 0-8 total
#SBATCH -x dgx001,dgx002


module load openmind8/anaconda/3-2022.10
export HDF5_USE_FILE_LOCKING=FALSE

source activate /om2/user/imgriff/conda_envs/pytorch_2

python3 eval_cue_duration_mono.py --config config/binaural_attn/word_task_25p_loc_v07_LN_last_valid_time_no_affine.yaml \
                 --ckpt_pat attn_cue_models/word_task_25p_loc_v07_LN_last_valid_time_no_affine/checkpoints/epoch=3-step=49432.ckpt \
                 --test_manifest_path binaural_test_manifests/cue_duration_test_manifest_1talker_only_0dB_snr.pkl \
                 --array_id $SLURM_ARRAY_TASK_ID \
                 --n_jobs 4 --exp_dir cue_duration_eval/ \

