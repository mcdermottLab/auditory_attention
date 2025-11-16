#!/bin/bash -l
#SBATCH --job-name=eval_model
#SBATCH --output=outLogs/sim_human_threshold_exmpt_v02_all_archs_%A_%a.out
#SBATCH --error=outLogs/sim_human_threshold_exmpt_v02_all_archs_%A_%a.err
#SBATCH --mem=8Gb
#SBATCH --cpus-per-task=4
#SBATCH --time=3:00:00
#SBATCH --partition=use-everything
#SBATCH --gres=gpu:1 --constraint=20GB
#SBATCH --array=0-59 #0-59 for all azimuth conditions 
#SBATCH -x dgx001,dgx002

module load openmind8/anaconda/3-2022.10
export HDF5_USE_FILE_LOCKING=FALSE

source activate /om2/user/imgriff/conda_envs/pytorch_2

which python3
rm -r /tmp/torchinductor_imgriff

python3 eval_sim_array_threshold_experiment_v02.py --config config/binaural_attn/word_task_v10_main_feature_gain_config.yaml \
                 --ckpt_path attn_cue_models/word_task_v10_main_feature_gain_config/checkpoints/epoch=1-step=24679-v1.ckpt \
                 --test_manifest binaural_test_manifests/sim_human_threshold_experiment_v02_min_reverb_room1006_30dB_bg_noise.pkl \
                 --location_idx $SLURM_ARRAY_TASK_ID \
                 --gpus 1 --n_jobs 4 --exp_dir binaural_eval/simulate_2024_human_threshold_experiment_v02_30_dB_pink_noise_min_verb_mit46_1004 \
                 --cue_type voice_and_location --no-overwrite --n_per_job 1 --sim_human_array_exmpt --run_all_stim --pink_noise_distractor

rm -r /tmp/torchinductor_imgriff
python3 eval_sim_array_threshold_experiment_v02.py --config config/arch_search/word_task_v10_4MGB_ln_first_arch_1.yaml \
                 --ckpt_path attn_cue_models/word_task_v10_4MGB_ln_first_arch_1/checkpoints/epoch=2-step=44750-v4.ckpt \
                 --test_manifest binaural_test_manifests/sim_human_threshold_experiment_v02_min_reverb_room1006_30dB_bg_noise.pkl \
                 --location_idx $SLURM_ARRAY_TASK_ID \
                 --gpus 1 --n_jobs 4 --exp_dir binaural_eval/simulate_2024_human_threshold_experiment_v02_30_dB_pink_noise_min_verb_mit46_1004 \
                 --cue_type voice_and_location --no-overwrite --n_per_job 1 --sim_human_array_exmpt --run_all_stim --pink_noise_distractor

rm -r /tmp/torchinductor_imgriff
python3 eval_sim_array_threshold_experiment_v02.py --config config/arch_search/word_task_v10_4MGB_ln_first_arch_2.yaml \
                 --ckpt_path attn_cue_models/word_task_v10_4MGB_ln_first_arch_2/checkpoints/epoch=3-step=50037-v3.ckpt \
                 --test_manifest binaural_test_manifests/sim_human_threshold_experiment_v02_min_reverb_room1006_30dB_bg_noise.pkl \
                 --location_idx $SLURM_ARRAY_TASK_ID \
                 --gpus 1 --n_jobs 4 --exp_dir binaural_eval/simulate_2024_human_threshold_experiment_v02_30_dB_pink_noise_min_verb_mit46_1004 \
                 --cue_type voice_and_location --no-overwrite --n_per_job 1 --sim_human_array_exmpt --run_all_stim --pink_noise_distractor

rm -r /tmp/torchinductor_imgriff
python3 eval_sim_array_threshold_experiment_v02.py --config config/arch_search/word_task_v10_4MGB_ln_first_arch_4.yaml \
                 --ckpt_path attn_cue_models/word_task_v10_4MGB_ln_first_arch_4/checkpoints/epoch=1-step=34375-v3.ckpt \
                 --test_manifest binaural_test_manifests/sim_human_threshold_experiment_v02_min_reverb_room1006_30dB_bg_noise.pkl \
                 --location_idx $SLURM_ARRAY_TASK_ID \
                 --gpus 1 --n_jobs 4 --exp_dir binaural_eval/simulate_2024_human_threshold_experiment_v02_30_dB_pink_noise_min_verb_mit46_1004 \
                 --cue_type voice_and_location --no-overwrite --n_per_job 1 --sim_human_array_exmpt --run_all_stim --pink_noise_distractor

rm -r /tmp/torchinductor_imgriff
python3 eval_sim_array_threshold_experiment_v02.py --config config/arch_search/word_task_v10_4MGB_ln_first_arch_6.yaml \
                 --ckpt_path attn_cue_models/word_task_v10_4MGB_ln_first_arch_6/checkpoints/epoch=3-step=42037.ckpt \
                 --test_manifest binaural_test_manifests/sim_human_threshold_experiment_v02_min_reverb_room1006_30dB_bg_noise.pkl \
                 --location_idx $SLURM_ARRAY_TASK_ID \
                 --gpus 1 --n_jobs 4 --exp_dir binaural_eval/simulate_2024_human_threshold_experiment_v02_30_dB_pink_noise_min_verb_mit46_1004 \
                 --cue_type voice_and_location --no-overwrite --n_per_job 1 --sim_human_array_exmpt --run_all_stim --pink_noise_distractor

rm -r /tmp/torchinductor_imgriff
python3 eval_sim_array_threshold_experiment_v02.py --config config/arch_search/word_task_v10_4MGB_ln_first_arch_7.yaml \
                 --ckpt_path attn_cue_models/word_task_v10_4MGB_ln_first_arch_7/checkpoints/epoch=0-step=8000-v3.ckpt \
                 --test_manifest binaural_test_manifests/sim_human_threshold_experiment_v02_min_reverb_room1006_30dB_bg_noise.pkl \
                 --location_idx $SLURM_ARRAY_TASK_ID \
                 --gpus 1 --n_jobs 4 --exp_dir binaural_eval/simulate_2024_human_threshold_experiment_v02_30_dB_pink_noise_min_verb_mit46_1004 \
                 --cue_type voice_and_location --no-overwrite --n_per_job 1 --sim_human_array_exmpt --run_all_stim --pink_noise_distractor

rm -r /tmp/torchinductor_imgriff
python3 eval_sim_array_threshold_experiment_v02.py --config config/arch_search/word_task_v10_4MGB_ln_first_arch_8.yaml \
                 --ckpt_path attn_cue_models/word_task_v10_4MGB_ln_first_arch_8/checkpoints/epoch=4-step=54716-v4.ckpt \
                 --test_manifest binaural_test_manifests/sim_human_threshold_experiment_v02_min_reverb_room1006_30dB_bg_noise.pkl \
                 --location_idx $SLURM_ARRAY_TASK_ID \
                 --gpus 1 --n_jobs 4 --exp_dir binaural_eval/simulate_2024_human_threshold_experiment_v02_30_dB_pink_noise_min_verb_mit46_1004 \
                 --cue_type voice_and_location --no-overwrite --n_per_job 1 --sim_human_array_exmpt --run_all_stim --pink_noise_distractor

rm -r /tmp/torchinductor_imgriff
python3 eval_sim_array_threshold_experiment_v02.py --config config/arch_search/word_task_v10_4MGB_ln_first_arch_9.yaml \
                 --ckpt_path attn_cue_models/word_task_v10_4MGB_ln_first_arch_9/checkpoints/epoch=1-step=16679-v4.ckpt \
                 --test_manifest binaural_test_manifests/sim_human_threshold_experiment_v02_min_reverb_room1006_30dB_bg_noise.pkl \
                 --location_idx $SLURM_ARRAY_TASK_ID \
                 --gpus 1 --n_jobs 4 --exp_dir binaural_eval/simulate_2024_human_threshold_experiment_v02_30_dB_pink_noise_min_verb_mit46_1004 \
                 --cue_type voice_and_location --no-overwrite --n_per_job 1 --sim_human_array_exmpt --run_all_stim --pink_noise_distractor

rm -r /tmp/torchinductor_imgriff
python3 eval_sim_array_threshold_experiment_v02.py --config config/arch_search/word_task_v10_4MGB_ln_first_arch_10.yaml \
                 --ckpt_path attn_cue_models/word_task_v10_4MGB_ln_first_arch_10/checkpoints/epoch=2-step=33358-v6.ckpt \
                 --test_manifest binaural_test_manifests/sim_human_threshold_experiment_v02_min_reverb_room1006_30dB_bg_noise.pkl \
                 --location_idx $SLURM_ARRAY_TASK_ID \
                 --gpus 1 --n_jobs 4 --exp_dir binaural_eval/simulate_2024_human_threshold_experiment_v02_30_dB_pink_noise_min_verb_mit46_1004 \
                 --cue_type voice_and_location --no-overwrite --n_per_job 1 --sim_human_array_exmpt --run_all_stim --pink_noise_distractor

rm -r /tmp/torchinductor_imgriff
python3 eval_sim_array_threshold_experiment_v02.py --config config/arch_search/word_task_v10_4MGB_ln_first_arch_12.yaml \
                 --ckpt_path attn_cue_models/word_task_v10_4MGB_ln_first_arch_12/checkpoints/epoch=2-step=33358-v4.ckpt \
                 --test_manifest binaural_test_manifests/sim_human_threshold_experiment_v02_min_reverb_room1006_30dB_bg_noise.pkl \
                 --location_idx $SLURM_ARRAY_TASK_ID \
                 --gpus 1 --n_jobs 4 --exp_dir binaural_eval/simulate_2024_human_threshold_experiment_v02_30_dB_pink_noise_min_verb_mit46_1004 \
                 --cue_type voice_and_location --no-overwrite --n_per_job 1 --sim_human_array_exmpt --run_all_stim --pink_noise_distractor


