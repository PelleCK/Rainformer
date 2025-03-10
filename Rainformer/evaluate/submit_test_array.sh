#!/bin/bash
source .env
export $(grep -v '^#' .env | xargs)

# Check if the required job name argument is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <job_name>"
    exit 1
fi

# Set the job name and directories based on the provided argument
export JOB_NAME=$1
LOG_DIR="${GENERAL_LOG_DIR}/${JOB_NAME}"

# Check if the log directory exists, and if not, create it
if [ ! -d "$LOG_DIR" ]; then
    mkdir -p "$LOG_DIR"
fi

echo "Submitting test array job with name $JOB_NAME"

# Submit the SLURM array job with the specified job name
sbatch --job-name=$JOB_NAME --export=ALL test_array.sh