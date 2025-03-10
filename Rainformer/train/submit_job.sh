#!/bin/bash
source .env
export $(grep -v '^#' .env | xargs)

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <job_name> [wandb_api_key]"
    exit 1
fi

JOB_NAME=$1
WANDB_API_KEY=$2
MODEL_DIR=${GENERAL_MODEL_DIR}/${JOB_NAME}/
LOG_DIR=${GENERAL_LOG_DIR}/${JOB_NAME}

if [ -z "$WANDB_API_KEY" ]; then
    echo "No WANDB_API_KEY provided. Logging will be disabled. Do you want to continue? (yes/no)"
    read answer
    if [ "$answer" != "yes" ]; then
        echo "Exiting to avoid unintentional job submission."
        exit 1
    fi
else
    export WANDB_API_KEY=$WANDB_API_KEY
fi

if [ ! -d "$LOG_DIR" ]; then
    mkdir -p "$LOG_DIR"
fi

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

# export JOB_NAME=$JOB_NAME
# export SLURM_JOB_NAME=$JOB_NAME

sbatch --job-name=$JOB_NAME --export=ALL train.sh
