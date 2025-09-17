#!/bin/bash -l 
#SBATCH --job-name=svm_fit_embedding
#SBATCH --output=outLogs/svm_fit_embedding_%A_%a.out
#SBATCH --error=outLogs/svm_fit_embedding_%A_%a.err
#SBATCH --mem=16Gb
#SBATCH --cpus-per-task=4
#SBATCH --time=1:00:00
#SBATCH --partition=normal
#SBATCH --array=1-26 # 27 jobs (9 layers x 3 targets)


module load openmind8/anaconda/3-2022.10
export HDF5_USE_FILE_LOCKING=FALSE

source activate /om2/user/imgriff/conda_envs/pytorch_2

# Configuration (fixed across array)
embedding_type="time_avg"
signal_idx=1
attn_idx=0

# Targets and layers
TARGETS=("location" "f0_bin" "word_class")
NUM_TARGETS=${#TARGETS[@]}
NUM_LAYERS=9

# Derive indices from SLURM task id
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
layer_idx=$(( TASK_ID % NUM_LAYERS ))
target_idx=$(( TASK_ID / NUM_LAYERS ))

# Safety checks
if (( target_idx < 0 || target_idx >= NUM_TARGETS )); then
  echo "Invalid target_idx ${target_idx} derived from TASK_ID ${TASK_ID}" >&2
  exit 1
fi

target="${TARGETS[$target_idx]}"

# Unique model output path
out_dir="models/svm"
mkdir -p "$out_dir"
model_name="${embedding_type}_${target}_layer${layer_idx}_signal${signal_idx}_attn${attn_idx}.pkl"
output_model="${out_dir}/${model_name}"

echo "Running task ${TASK_ID} -> layer_idx=${layer_idx}, target=${target}"

python3 run_fit_embedding_svm.py \
  --embedding_type "${embedding_type}" \
  --layer_idx "${layer_idx}" \
  --signal_idx "${signal_idx}" \
  --attn_idx "${attn_idx}" \
  --target "${target}" \
  --output_model "${output_model}"
