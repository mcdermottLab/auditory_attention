#!/bin/bash -l
#SBATCH --job-name=eval_model
#SBATCH --output=outLogs/sim_human_array_exmpt_v02_alt_archs_%A_%a.out
#SBATCH --error=outLogs/sim_human_array_exmpt_v02_alt_archs_%A_%a.err
#SBATCH --mem=8Gb
#SBATCH --cpus-per-task=2
#SBATCH --time=1:00:00
#SBATCH --partition=use-everything
#SBATCH --gres=gpu:1 --constraint=20GB
#SBATCH --array=0-419 #0-119;  0-419 for min reverb room  manifest 
#SBATCH -x dgx001,dgx002

module load openmind8/anaconda/3-2022.10
export HDF5_USE_FILE_LOCKING=FALSE

source activate /om2/user/imgriff/conda_envs/pytorch_2

which python3
rm -r /tmp/torchinductor_imgriff

python3 eval_symmetric_distractors.py --config config/arch_search/binaural_attn/word_task_v09_4MGB_ln_first_arch_1.yaml \
                 --ckpt_path attn_cue_models/word_task_v09_4MGB_ln_first_arch_1/checkpoints/epoch=2-step=40080.ckpt \
                 --test_manifest binaural_test_manifests/human_array_exmpt_sim_w_front_back_21_to_6_dBSNR_min_reverb_mit_room_v02.pkl \
                 --location_idx $SLURM_ARRAY_TASK_ID \
                 --gpus 1 --n_jobs 2 --exp_dir binaural_eval/simulate_2024_human_array_experiment_v02 \
                 --cue_type voice_and_location --no-overwrite --n_per_job 1 --sim_human_array_exmpt --run_all_stim 

rm -r /tmp/torchinductor_imgriff

python3 eval_symmetric_distractors.py --config config/arch_search/binaural_attn/word_task_v09_4MGB_ln_first_arch_2.yaml \
                 --ckpt_path attn_cue_models/word_task_v09_4MGB_ln_first_arch_2/checkpoints/epoch=2-step=32806-v4.ckpt \
                 --test_manifest binaural_test_manifests/human_array_exmpt_sim_w_front_back_21_to_6_dBSNR_min_reverb_mit_room_v02.pkl \
                 --location_idx $SLURM_ARRAY_TASK_ID \
                 --gpus 1 --n_jobs 2 --exp_dir binaural_eval/simulate_2024_human_array_experiment_v02 \
                 --cue_type voice_and_location --no-overwrite --n_per_job 1 --sim_human_array_exmpt --run_all_stim 

rm -r /tmp/torchinductor_imgriff
python3 eval_symmetric_distractors.py --config config/arch_search/binaural_attn/word_task_v09_4MGB_ln_first_arch_4.yaml \
                 --ckpt_path attn_cue_models/word_task_v09_4MGB_ln_first_arch_4/checkpoints/epoch=0-step=8000-v4.ckpt \
                 --test_manifest binaural_test_manifests/human_array_exmpt_sim_w_front_back_21_to_6_dBSNR_min_reverb_mit_room_v02.pkl \
                 --location_idx $SLURM_ARRAY_TASK_ID \
                 --gpus 1 --n_jobs 2 --exp_dir binaural_eval/simulate_2024_human_array_experiment_v02 \
                 --cue_type voice_and_location --no-overwrite --n_per_job 1 --sim_human_array_exmpt --run_all_stim 

rm -r /tmp/torchinductor_imgriff
python3 eval_symmetric_distractors.py --config config/arch_search/binaural_attn/word_task_v09_4MGB_ln_first_arch_6.yaml \
                 --ckpt_path attn_cue_models/word_task_v09_4MGB_ln_first_arch_6/checkpoints/epoch=2-step=32806-v3.ckpt \
                 --test_manifest binaural_test_manifests/human_array_exmpt_sim_w_front_back_21_to_6_dBSNR_min_reverb_mit_room_v02.pkl \
                 --location_idx $SLURM_ARRAY_TASK_ID \
                 --gpus 1 --n_jobs 2 --exp_dir binaural_eval/simulate_2024_human_array_experiment_v02 \
                 --cue_type voice_and_location --no-overwrite --n_per_job 1 --sim_human_array_exmpt --run_all_stim 

rm -r /tmp/torchinductor_imgriff
python3 eval_symmetric_distractors.py --config config/arch_search/binaural_attn/word_task_v09_4MGB_ln_first_arch_7.yaml \
                 --ckpt_path attn_cue_models/word_task_v09_4MGB_ln_first_arch_7/checkpoints/epoch=0-step=8000-v7.ckpt \
                 --test_manifest binaural_test_manifests/human_array_exmpt_sim_w_front_back_21_to_6_dBSNR_min_reverb_mit_room_v02.pkl \
                 --location_idx $SLURM_ARRAY_TASK_ID \
                 --gpus 1 --n_jobs 2 --exp_dir binaural_eval/simulate_2024_human_array_experiment_v02 \
                 --cue_type voice_and_location --no-overwrite --n_per_job 1 --sim_human_array_exmpt --run_all_stim 

rm -r /tmp/torchinductor_imgriff
python3 eval_symmetric_distractors.py --config config/arch_search/binaural_attn/word_task_v09_4MGB_ln_first_arch_8.yaml \
                 --ckpt_path attn_cue_models/word_task_v09_4MGB_ln_first_arch_8/checkpoints/epoch=0-step=4000.ckpt \
                 --test_manifest binaural_test_manifests/human_array_exmpt_sim_w_front_back_21_to_6_dBSNR_min_reverb_mit_room_v02.pkl \
                 --location_idx $SLURM_ARRAY_TASK_ID \
                 --gpus 1 --n_jobs 2 --exp_dir binaural_eval/simulate_2024_human_array_experiment_v02 \
                 --cue_type voice_and_location --no-overwrite --n_per_job 1 --sim_human_array_exmpt --run_all_stim 

rm -r /tmp/torchinductor_imgriff
python3 eval_symmetric_distractors.py --config config/arch_search/binaural_attn/word_task_v09_4MGB_ln_first_arch_9.yaml \
                 --ckpt_path attn_cue_models/word_task_v09_4MGB_ln_first_arch_9/checkpoints/epoch=0-step=4000-v5.ckpt \
                 --test_manifest binaural_test_manifests/human_array_exmpt_sim_w_front_back_21_to_6_dBSNR_min_reverb_mit_room_v02.pkl \
                 --location_idx $SLURM_ARRAY_TASK_ID \
                 --gpus 1 --n_jobs 2 --exp_dir binaural_eval/simulate_2024_human_array_experiment_v02 \
                 --cue_type voice_and_location --no-overwrite --n_per_job 1 --sim_human_array_exmpt --run_all_stim 
