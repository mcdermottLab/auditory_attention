#!/bin/bash -l 
#SBATCH --job-name=eval_cue_duration
#SBATCH --output=outLogs/cue_duration_test_stim_%A_%a.out
#SBATCH --error=outLogs/cue_duration_test_stim_%A_%a.err
#SBATCH --mem=4Gb
#SBATCH --cpus-per-task=4
#SBATCH --time=0:20:00
#SBATCH --partition=normal
#SBATCH --gres=gpu:1 --constraint=20GB
#SBATCH --array=0-8 # 0-8 total
#SBATCH -x dgx001,dgx002


module load openmind8/anaconda/3-2022.10
export HDF5_USE_FILE_LOCKING=FALSE

source activate /om2/user/imgriff/conda_envs/pytorch_2

python3 eval_cue_duration_mono.py --config /om2/user/imgriff/projects/torch_2_aud_attn/config/binaural_attn/word_task_half_co_loc_v07.yaml \
                 --ckpt_pat /om2/user/imgriff/projects/torch_2_aud_attn/attn_cue_models/word_task_half_co_loc_v07/checkpoints/epoch=2-step=46074.ckpt \
                 --test_manifest_path /om2/user/imgriff/projects/torch_2_aud_attn/binaural_test_manifests/cue_duration_test_manifest_1talker_only_0dB_snr.pkl \
                 --array_id $SLURM_ARRAY_TASK_ID \
                 --n_jobs 4 --exp_dir cue_duration_eval/ \

