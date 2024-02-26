#!/bin/bash -l
#SBATCH --job-name=eval_model
#SBATCH --output=outLogs/eval_binaural_word_task_half_co_loc_v07_%A_%a.out
#SBATCH --error=outLogs/eval_binaural_word_task_half_co_loc_v07_%A_%a.err
#SBATCH --mem=10Gb
#SBATCH --cpus-per-task=2
#SBATCH --time=2:00:00
#SBATCH --partition=use-everything
#SBATCH --gres=gpu:1 --constraint=20GB
#SBATCH --array=1-46 # 0-46 for full
#SBATCH -x dgx001,dgx002

module load openmind8/anaconda/3-2022.10
export HDF5_USE_FILE_LOCKING=FALSE

source activate /om2/user/imgriff/conda_envs/pytorch_2

which python3
python3 eval_binaural_w_manifest.py --config config/binaural_attn/word_task_half_co_loc_v07.yaml \
                 --ckpt_path attn_cue_models/word_task_half_co_loc_v07/checkpoints/epoch=2-step=46074.ckpt \
                 --location_manifest binaural_test_manifests/front_back_conditions.pkl \
                 --model_name word_task_half_co_loc_v07 --location_idx $SLURM_ARRAY_TASK_ID \
                 --gpus 1 --n_jobs 2 --exp_dir binaural_eval/front_back_conditions \
                 --cue_type voice_and_location --no-overwrite

