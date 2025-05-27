#!/bin/bash -l
#SBATCH --job-name=eval_model
#SBATCH --output=outLogs/eval_precedence_alt_archs_%A_%a.out
#SBATCH --error=outLogs/eval_precedence_alt_archs_%A_%a.err
#SBATCH --mem=10Gb
#SBATCH --cpus-per-task=2
#SBATCH --time=3:00:00
#SBATCH --partition=use-everything
#SBATCH --gres=gpu:1 --constraint=20GB
#SBATCH --array=0-29 # 0-29 if running 5 per job 
#SBATCH -x dgx001,dgx002

module load openmind8/anaconda/3-2022.10
export HDF5_USE_FILE_LOCKING=FALSE

source activate /om2/user/imgriff/conda_envs/pytorch_2
#                  --test_manifest binaural_test_manifests/precedence_distractor_conditions_co_loc_conditions.pkl \

which python3

rm -rf /tmp/torchinductor_imgriff
python3 eval_precedence.py  --config config/binaural_attn/word_task_v10_main_feature_gain_config.yaml \
                 --ckpt_path attn_cue_models/word_task_v10_main_feature_gain_config/checkpoints/epoch=1-step=24679-v1.ckpt \
                 --test_manifest binaural_test_manifests/freymen_1999_test_conds.pkl \
                 --model_name word_task_v10_main_feature_gain_config --location_idx $SLURM_ARRAY_TASK_ID \
                 --gpus 1 --n_jobs 2 --exp_dir binaural_eval/precedence_distractor_test_v02 \
                 --cue_type voice_and_location --no-overwrite --n_per_job 1

rm -rf /tmp/torchinductor_imgriff
python3 eval_precedence.py --config config/arch_search/word_task_v10_4MGB_ln_first_arch_1.yaml \
                 --ckpt_path attn_cue_models/word_task_v10_4MGB_ln_first_arch_1/checkpoints/epoch=2-step=44750-v4.ckpt \
                 --test_manifest binaural_test_manifests/freymen_1999_test_conds.pkl \
                 --model_name word_task_v10_4MGB_ln_first_arch_1 --location_idx $SLURM_ARRAY_TASK_ID \
                 --gpus 1 --n_jobs 2 --exp_dir binaural_eval/precedence_distractor_test_v02 \
                 --cue_type voice_and_location --no-overwrite --n_per_job 1

rm -rf /tmp/torchinductor_imgriff
python3 eval_precedence.py --config config/arch_search/word_task_v10_4MGB_ln_first_arch_2.yaml \
                 --ckpt_path attn_cue_models/word_task_v10_4MGB_ln_first_arch_2/checkpoints/epoch=3-step=50037-v3.ckpt \
                 --test_manifest binaural_test_manifests/freymen_1999_test_conds.pkl \
                 --model_name word_task_v10_4MGB_ln_first_arch_2 --location_idx $SLURM_ARRAY_TASK_ID \
                 --gpus 1 --n_jobs 2 --exp_dir binaural_eval/precedence_distractor_test_v02 \
                 --cue_type voice_and_location --no-overwrite --n_per_job 1

rm -r /tmp/torchinductor_imgriff
python3 eval_precedence.py --config config/arch_search/word_task_v10_4MGB_ln_first_arch_4.yaml \
                 --ckpt_path attn_cue_models/word_task_v10_4MGB_ln_first_arch_4/checkpoints/epoch=1-step=34375-v3.ckpt \
                 --test_manifest binaural_test_manifests/freymen_1999_test_conds.pkl \
                 --model_name word_task_v10_4MGB_ln_first_arch_4 --location_idx $SLURM_ARRAY_TASK_ID \
                 --gpus 1 --n_jobs 2 --exp_dir binaural_eval/precedence_distractor_test_v02 \
                 --cue_type voice_and_location --no-overwrite --n_per_job 1

rm -r /tmp/torchinductor_imgriff
python3 eval_precedence.py --config config/arch_search/word_task_v10_4MGB_ln_first_arch_6.yaml \
                 --ckpt_path attn_cue_models/word_task_v10_4MGB_ln_first_arch_6/checkpoints/epoch=3-step=42037.ckpt \
                 --test_manifest binaural_test_manifests/freymen_1999_test_conds.pkl \
                 --model_name word_task_v10_4MGB_ln_first_arch_6 --location_idx $SLURM_ARRAY_TASK_ID \
                 --gpus 1 --n_jobs 2 --exp_dir binaural_eval/precedence_distractor_test_v02 \
                 --cue_type voice_and_location --no-overwrite --n_per_job 1

rm -r /tmp/torchinductor_imgriff
python3 eval_precedence.py --config config/arch_search/word_task_v10_4MGB_ln_first_arch_7.yaml \
                 --ckpt_path attn_cue_models/word_task_v10_4MGB_ln_first_arch_7/checkpoints/epoch=0-step=8000-v3.ckpt \
                 --test_manifest binaural_test_manifests/freymen_1999_test_conds.pkl \
                 --model_name word_task_v10_4MGB_ln_first_arch_7 --location_idx $SLURM_ARRAY_TASK_ID \
                 --gpus 1 --n_jobs 2 --exp_dir binaural_eval/precedence_distractor_test_v02 \
                 --cue_type voice_and_location --no-overwrite --n_per_job 1

rm -r /tmp/torchinductor_imgriff
python3 eval_precedence.py --config config/arch_search/word_task_v10_4MGB_ln_first_arch_8.yaml \
                 --ckpt_path attn_cue_models/word_task_v10_4MGB_ln_first_arch_8/checkpoints/epoch=4-step=54716-v4.ckpt \
                 --test_manifest binaural_test_manifests/freymen_1999_test_conds.pkl \
                 --model_name word_task_v10_4MGB_ln_first_arch_8 --location_idx $SLURM_ARRAY_TASK_ID \
                 --gpus 1 --n_jobs 2 --exp_dir binaural_eval/precedence_distractor_test_v02 \
                 --cue_type voice_and_location --no-overwrite --n_per_job 1

rm -rf /tmp/torchinductor_imgriff

python3 eval_precedence.py --config config/arch_search/word_task_v10_4MGB_ln_first_arch_9.yaml \
                 --ckpt_path attn_cue_models/word_task_v10_4MGB_ln_first_arch_9/checkpoints/epoch=1-step=16679-v4.ckpt \
                 --test_manifest binaural_test_manifests/freymen_1999_test_conds.pkl \
                 --model_name word_task_v10_4MGB_ln_first_arch_9 --location_idx $SLURM_ARRAY_TASK_ID \
                 --gpus 1 --n_jobs 2 --exp_dir binaural_eval/precedence_distractor_test_v02 \
                 --cue_type voice_and_location --no-overwrite --n_per_job 1

rm -fr /tmp/torchinductor_imgriff

python3 eval_precedence.py --config config/arch_search/word_task_v10_4MGB_ln_first_arch_10.yaml \
                 --ckpt_path attn_cue_models/word_task_v10_4MGB_ln_first_arch_10/checkpoints/epoch=2-step=33358-v6.ckpt \
                 --test_manifest binaural_test_manifests/freymen_1999_test_conds.pkl \
                 --model_name word_task_v10_4MGB_ln_first_arch_10 --location_idx $SLURM_ARRAY_TASK_ID \
                 --gpus 1 --n_jobs 2 --exp_dir binaural_eval/precedence_distractor_test_v02 \
                 --cue_type voice_and_location --no-overwrite --n_per_job 1

rm -fr /tmp/torchinductor_imgriff

python3 eval_precedence.py --config config/arch_search/word_task_v10_4MGB_ln_first_arch_12.yaml \
                 --ckpt_path attn_cue_models/word_task_v10_4MGB_ln_first_arch_12/checkpoints/epoch=2-step=33358-v4.ckpt \
                 --test_manifest binaural_test_manifests/freymen_1999_test_conds.pkl \
                 --model_name word_task_v10_4MGB_ln_first_arch_12 --location_idx $SLURM_ARRAY_TASK_ID \
                 --gpus 1 --n_jobs 2 --exp_dir binaural_eval/precedence_distractor_test_v02 \
                 --cue_type voice_and_location --no-overwrite --n_per_job 1

