#!/bin/bash
#SBATCH --account=das
#SBATCH --partition=das
#SBATCH --qos=das-large
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:rtx_3090:4
#SBATCH --time=2-00:00:00
#SBATCH --mem=20G
#TODO: adjust to your log directory path
#SBATCH --output=/path/to/logs/%x/%x-%j.out
#SBATCH --error=/path/to/logs/%x/%x-%j.err
#TODO: add your email address if you want to receive notifications
#SBATCH --mail-type=ALL
#SBATCH --mail-user=

batch_size=16
N_GPUS=$SLURM_NTASKS
effective_batch_size=$((N_GPUS * batch_size))
echo "N_GPUS: $N_GPUS, effective_batch_size: $effective_batch_size"

export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)
export MASTER_PORT=$(shuf -i 30000-40000 -n 1)
export WORLD_SIZE=$SLURM_NTASKS
echo "WORLD_SIZE: $WORLD_SIZE, MASTER_ADDR: $MASTER_ADDR, MASTER_PORT: $MASTER_PORT"

#TODO: use config file for these
model_dir=/vol/knmimo-nobackup/users/pkools/thesis-forecasting/Rainformer/results/model_checkpoints/${SLURM_JOB_NAME}/
train_ids_fn='/vol/knmimo-nobackup/restore/knmimo/thesis_pelle/data/preprocessed/data_splits/ldcast_train_200922_avg001mm_4x20y.npy' # ldcast_val_2008_avg001mm_4x20y.npy
val_ids_fn='/vol/knmimo-nobackup/restore/knmimo/thesis_pelle/data/preprocessed/data_splits/ldcast_val_2008_avg001mm_4x20y.npy'

#TODO: put this in python script, add boolean argument adjust_lr
# the loss is not scaled w.r.t sequence length and batch size, so we need to adjust the learning rate
lr_orig=0.0001
base_batch_size=16
y_seq_size_orig=9
y_seq_size=20
# how much the sequence length has increased
seq_increase=$(echo "scale=6; $y_seq_size / $y_seq_size_orig" | bc)
# how much the batch size has increased
batch_increase=$(echo "scale=6; $effective_batch_size / $base_batch_size" | bc)

echo "seq_increase: $seq_increase, batch_increase: $batch_increase"

# the learning rate should be scaled down proportionally
lr=$(echo "scale=8; $lr_orig / ($seq_increase * $batch_increase)" | bc)

printf "Adjusted learning rate: %.8f\n" "$lr"

#TODO: change to your virtual environment path
source /vol/knmimo-nobackup/users/pkools/thesis-forecasting/Rainformer/.rainformer_venv/bin/activate

srun --kill-on-bad-exit=1 python train_on_cluster_ddp.py \
    --x_seq_size=4 \
    --y_seq_size=$y_seq_size \
    --model_dir=$model_dir \
    --batch_size=$batch_size \
    --train_IDs_fn=$train_ids_fn \
    --val_IDs_fn=$val_ids_fn \
    --lr=$lr
