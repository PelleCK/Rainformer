#!/bin/bash

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <job_name> <wandb_api_key>"
    exit 1
fi

JOB_NAME=$1
WANDB_API_KEY=$2
#TODO: change to your model directory path
MODEL_DIR="/vol/knmimo-nobackup/users/pkools/thesis-forecasting/Rainformer/results/model_checkpoints/${JOB_NAME}/"
#TODO: change to your log directory path
LOG_DIR="/vol/knmimo-nobackup/users/pkools/thesis-forecasting/Rainformer/results/logs/${JOB_NAME}"


# Check if the log directory exists, and if not, create it
if [ ! -d "$LOG_DIR" ]; then
    mkdir -p "$LOG_DIR"
fi

# Check if model directory already exists
if [ -d "$MODEL_DIR" ]; then
    echo "Directory $MODEL_DIR already exists. Do you want to continue and possibly overwrite existing job? (yes/no)"
else
    echo "Directory $MODEL_DIR does not exist. Are you sure you want to create a new job with this name? (yes/no)"
fi
read answer
if [ "$answer" != "yes" ]; then
    echo "Exiting to avoid unintentional job submission."
    exit 1
fi

echo "Submitting job with name $JOB_NAME and model directory $MODEL_DIR"

export WANDB_API_KEY=$WANDB_API_KEY
export JOB_NAME=$JOB_NAME

# Submit the SLURM job
sbatch --job-name=$JOB_NAME --export=ALL train.sh