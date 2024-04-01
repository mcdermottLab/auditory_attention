#!/bin/bash -l 
#SBATCH --job-name=eval_unf_lang
#SBATCH --output=outLogs/binaural_unfamiliar_language_distractor_stim_%A_%a.out
#SBATCH --error=outLogs/binaural_unfamiliar_language_distractor_stim_%A_%a.err
#SBATCH --mem=12Gb
#SBATCH --cpus-per-task=4
#SBATCH --time=0:15:00
#SBATCH --partition=use-everything
#SBATCH --gres=gpu:1 --constraint=20GB
#SBATCH --array=4-6 # 0-17
#SBATCH -x dgx001,dgx002


module load openmind8/anaconda/3-2022.10
export HDF5_USE_FILE_LOCKING=FALSE

source activate /om2/user/imgriff/conda_envs/pytorch_2

python3 eval_unfamiliar_distractor_stim.py --config config/binaural_attn/word_task_half_co_loc_v07.yaml \
                 --stim_manifest_path /om/user/imgriff/datasets/human_distractor_language_2024/final_stim_manifest_w_cue_tg_lang_dists.pdpkl \
                 --test_manifest binaural_test_manifests/unfamiliar_distractor_language_1_distractor.pkl \
                 --ckpt_pat attn_cue_models/word_task_half_co_loc_v07/checkpoints/epoch=2-step=46074.ckpt \
                 --array_id $SLURM_ARRAY_TASK_ID \
                 --n_jobs 4 --exp_dir unfamiliar_distractor/ \

