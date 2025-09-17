#!/bin/bash -l
#SBATCH --job-name=jupyter_notebook
#SBATCH --output=outLogs/notebook_%j.out
#SBATCH --error=outLogs/notebook_%j.err
#SBATCH --mem=50G
#SBATCH --time=3:00:00
#SBATCH --partition=ou_bcs_high
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --nodes=1

# Load modules
module load miniforge

# Activate conda environment
source activate pytorch_2

# Set up Python path to include the project root
export PYTHONPATH="${PYTHONPATH}:/orcd/data/jhm/001/om2/rphess/projects/github.com/Auditory-Attention"

# Set up environment variables for distributed training (needed for Muon)
export MASTER_ADDR=localhost
export MASTER_PORT=29500

# Set CUDA device to use
export CUDA_VISIBLE_DEVICES=0

# Print environment info for debugging
echo "=== Environment Setup ==="
echo "Python path: $PYTHONPATH"
echo "Conda environment: $CONDA_DEFAULT_ENV"
echo "CUDA visible devices: $CUDA_VISIBLE_DEVICES"
echo "Working directory: $(pwd)"
echo "========================="

# Start Jupyter notebook
export LC_ALL=C
unset XDG_RUNTIME_DIR
jupyter notebook --no-browser --port=8888 --ip=0.0.0.0
