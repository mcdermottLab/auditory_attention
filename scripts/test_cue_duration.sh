#!/bin/bash -l 
#SBATCH --job-name=eval_cue_duration
#SBATCH --output=outLogs/cue_duration_test_stim_%A_%a.out
#SBATCH --error=outLogs/cue_duration_test_stim_%A_%a.err
#SBATCH --mem=4Gb
#SBATCH --cpus-per-task=4
#SBATCH --time=0:10:00
#SBATCH --partition=use-everything
#SBATCH --gres=gpu:1 --constraint=20GB
#SBATCH --array=0-7 # 0-8 total
#SBATCH -x dgx001,dgx002

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"



module load openmind8/anaconda/3-2022.10
export HDF5_USE_FILE_LOCKING=FALSE

source activate /om2/user/imgriff/conda_envs/pytorch_2


python3 -m src.eval_cue_duration --config /om2/user/imgriff/projects/torch_2_aud_attn/config/binaural_attn/word_task_v10_main_feature_gain_config.yaml \
                 --ckpt_path attn_cue_models/word_task_v10_main_feature_gain_config/checkpoints/epoch=1-step=24679-v1.ckpt \
                 --test_manifest_path /om2/user/imgriff/projects/torch_2_aud_attn/binaural_test_manifests/cue_duration_test_manifest_v02.pkl \
                 --array_id $SLURM_ARRAY_TASK_ID \
                 --n_jobs 4 --exp_dir cue_duration_eval_center_crop/ \

