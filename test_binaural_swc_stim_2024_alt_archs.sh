#!/bin/bash -l 
#SBATCH --job-name=eval_binaural_swc
#SBATCH --output=outLogs/binaural_swc_test_2024_v10_arch_search_%A_%a.out
#SBATCH --error=outLogs/binaural_swc_test_2024_v10_arch_search_%A_%a.err
#SBATCH --mem=12Gb
#SBATCH --cpus-per-task=4
#SBATCH --time=0:15:00
#SBATCH --partition=use-everything
#SBATCH --gres=gpu:1 --constraint=20GB
#SBATCH --array=183,193,423,424 # 0-548 for 9 archs on each test
#SBATCH -x dgx001,dgx002,node104

module load openmind8/anaconda/3-2022.10
export HDF5_USE_FILE_LOCKING=FALSE

source activate /om2/user/imgriff/conda_envs/pytorch_2

# sometimes get compilation issues - remove just to be safe
rm -r /tmp/torchinductor_imgriff


python3 eval_swc_mono_stim.py --config_list_path swc_test_manifests/arch_search_configs_v10_all_conds_w_latest_ckpts.pkl \
                 --array_id $SLURM_ARRAY_TASK_ID \
                 --n_jobs 4 --exp_dir swc_2024_eval_full_stim/ \
                 --stim_path /om/user/imgriff/datasets/human_word_rec_SWC_2024/model_eval_stim.h5 \
                 --stim_cond_map binaural_test_manifests/swc_all_cond_h5_job_manifest.pkl \
                 --full_h5_stim_set --overwrite
