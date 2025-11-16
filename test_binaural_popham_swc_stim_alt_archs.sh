#!/bin/bash -l 
#SBATCH --job-name=eval_popham_swc
#SBATCH --output=outLogs/binaural_popham_swc_conds_all_stim_alt_archs_v10_%A_%a.out
#SBATCH --error=outLogs/binaural_popham_swc_conds_all_stim_alt_archs_v10_%A_%a.err
#SBATCH --mem=12Gb
#SBATCH --cpus-per-task=4
#SBATCH --time=0:10:00
#SBATCH --partition=normal
#SBATCH --gres=gpu:1 --constraint=20GB
#SBATCH --array=0-107 # 0-107
#SBATCH -x dgx001,dgx002,node104

module load openmind8/anaconda/3-2022.10
export HDF5_USE_FILE_LOCKING=FALSE

source activate /om2/user/imgriff/conda_envs/pytorch_2

rm -rf /tmp/torchinductor_imgriff
python3 eval_swc_popham_2024.py --config_list_path swc_test_manifests/arch_search_configs_v10_all_popham_conds_w_latest_ckpts.pkl  \
                 --array_id $SLURM_ARRAY_TASK_ID \
                 --n_jobs 4 \
                 --stim_path /om/user/imgriff/datasets/human_swc_popham_exmpt_2024/model_eval_h5s/ \
                 --stim_cond_map all_stim_swc_popham_exmpt_2024_cond_manifest.pkl \
                 --exp_dir popham_swc_eval_all_stim/ \

