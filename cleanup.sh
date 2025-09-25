#!/bin/bash
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

# Define the directory containing your log files
LOG_DIR="/om2/user/rphess/Auditory-Attention/outLogs"

# Use the 'find' command to locate and delete files older than 10 days
find "$LOG_DIR" -type f \( -name "*.out" -o -name "*.err" \) -mtime +10 -delete
