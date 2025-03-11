#!/bin/bash
#SBATCH --account=icis
#SBATCH --partition=icis
#SBATCH --qos=icis-small
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:rtx_a6000:1
#SBATCH --time=4:00:00
#SBATCH --mem=20G
#SBATCH --output=${GENERAL_LOG_DIR}/%x/%x-%j.out
#SBATCH --error=${GENERAL_LOG_DIR}/%x/%x-%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=pelle.kools@ru.nl

export CUDA_LAUNCH_BLOCKING=1

# Set variables for testing
# batch_size=64
# test_ids_fn="${DATA_SPLITS_DIR}/list_IDs_202324_avg001mm_4x9y_test_rainformer.npy"
# # list_IDs_202324_avg001mm_9x9y_test_pretrained_rainformer.npy'
# # ldcast_test_202324_avg001mm_4x20y.npy'
# # MODEL_CHECKPOINT='/vol/knmimo-nobackup/users/pkools/thesis-forecasting/Rainformer/results/model_checkpoints/rainformer_ddp_preempt/best-epoch25-loss17971.9797.ckpt'
# data_path=${PREPROCESSED_RTCOR_DIR}
# # data_path=${PRETRAINED_DATA_DIR}
# # data_path=${RAINFORMER_NL_50_DIR}
# model_checkpoint=$MODEL_CHECKPOINT
CONFIG_FILE="./evaluation_configs/evaluation_config_4x9y_rtcor_from_9x9y_heavy.yaml"

source ${VENV_DIR}/bin/activate

# Run the test script
srun --export=ALL python test_on_cluster.py --config_path=$CONFIG_FILE
# srun --export=ALL python test_on_cluster.py \
#     --x_seq_size=4 \
#     --y_seq_size=9 \
#     --batch_size=$batch_size \
#     --test_IDs_fn=$test_ids_fn \
#     --checkpoint_path=$MODEL_CHECKPOINT \
#     --data_path=$data_path \
#     --window_size=8 \
#     --input_h_w="[256, 256]" \
#     --use_orig_data=False \
#     --undo_prep=True
#     # --evaluate_first_n=9
