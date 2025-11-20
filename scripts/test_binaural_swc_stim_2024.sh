#!/bin/bash -l 
#SBATCH --job-name=eval_binaural_swc
#SBATCH --output=outLogs/binaural_swc_test_2024_%A_%a.out
#SBATCH --error=outLogs/binaural_swc_test_2024_%A_%a.err
#SBATCH --mem=12Gb
#SBATCH --cpus-per-task=4
#SBATCH --time=0:10:00
#SBATCH --partition=use-everything
#SBATCH --gres=gpu:1 --constraint=20GB
#SBATCH --array=0-60 # 0-60 for standard test
#SBATCH -x dgx001,dgx002,node093

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"


module load openmind8/anaconda/3-2022.10
export HDF5_USE_FILE_LOCKING=FALSE

source activate /om2/user/imgriff/conda_envs/pytorch_2

# sometimes get compilation issues - remove just to be safe
rm -r /tmp/torchinductor_imgriff

python3 -m src.eval_swc_mono_stim --config config/binaural_attn/word_task_v10_main_feature_gain_config.yaml \
                 --ckpt_path attn_cue_models/word_task_v10_main_feature_gain_config/checkpoints/epoch=1-step=24679-v1.ckpt  \
                 --array_id $SLURM_ARRAY_TASK_ID \
                 --n_jobs 4 --exp_dir swc_2024_eval_full_stim/ \
                 --stim_path /om/user/imgriff/datasets/human_word_rec_SWC_2024/model_eval_stim.h5 \
                 --stim_cond_map binaural_test_manifests/swc_all_cond_h5_job_manifest.pkl \
                 --full_h5_stim_set


