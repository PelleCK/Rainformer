# Rainformer

Rainformer is a PyTorch-based encoder-forecaster model for precipitation nowcasting, as introduced in [Rainformer: Features Extraction Balanced Network for Radar-Based Precipitation Nowcasting](https://ieeexplore.ieee.org/abstract/document/9743916). This repository is a fork of the original Rainformer repository and includes adaptations to support my thesis research on **precipitation nowcasting**, specifically focusing on high-intensity precipitation prediction. My research focuses on the effect of input and output sequence length on the model's performance, as well as the impact of adding temperature data to the input features.

---

## Key Features

- **Rainformer**:
  - A precipitation nowcasting model combining local and global attention mechanisms.
  - Includes channel-attention and spatial-attention modules for effective feature extraction.

- **Contributions**:
  - Customizations to train and evaluate Rainformer with our data setup.
  - Add possibility to train on a multi-gpu cluster environment with SLURM.
  - Support for logging with Weights & Biases.
  - Additional evaluation metrics for precipitation nowcasting.

---

## Requirements

This repository uses **Python 3.10.12**. All other dependencies are listed in `requirements.txt`. To set up your environment:

1. Create and activate a Python virtual environment:
   ```
   python -m venv rainformer_env
   source rainformer_env/bin/activate
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

---

## Data


---

## Training the Model

This repository supports both **single-GPU** and **multi-GPU** training setups. For cluster environments with **SLURM**, refer to the instructions below.

### Training on SLURM

To submit a training job on SLURM, go into the `Rainformer/train` directory and run the following command:
```
./submit_job.sh [job_name] [wandb_api_key]
```
Here, you must add a `job_name`, as it will be used to create directories for logging and model checkpoints. The `wandb_api_key` is optional and can be used to log training metrics with Weights & Biases. If not provided, the script will skip logging.

- The [`submit_job.sh`](./Rainformer/train/submit_job.sh) script:
  - Automatically sets up logging and model checkpoint directories.
  - Exports the Weights & Biases API key for logging.
  - Submits the [`train.sh`](./Rainformer/train.sh) script to SLURM using `sbatch`.

- The [`train.sh`](./Rainformer/train.sh) script:
  - Configures SLURM-specific parameters (e.g., GPUs, CPUs, memory).
  - Defines the Distributed Data Parallel (DDP) settings (`MASTER_ADDR`, `MASTER_PORT`, `WORLD_SIZE`).
  - defines the config file path
  - Runs [`train_on_cluster_ddp.py`](./Rainformer/train_on_cluster_ddp.py) with the specified configuration.

A typical configuration file is provided in [`train_config.yaml`](./Rainformer/train/train_configs/config.yaml). You can modify the configuration file to adjust the training parameters.

### Training Locally on a Single GPU

For a single GPU setup without SLURM, you can run the training script directly:
```
python train_on_cluster_ddp.py \
  --x_seq_size=4 \
  --y_seq_size=20 \
  --batch_size=16 \
  --train_IDs_fn=[path_to_train_data] \
  --val_IDs_fn=[path_to_val_data] \
  --model_dir=[path_to_save_model]
```

Ensure the following for local training:
- Set `world_size=1` and `rank=0` in the configuration.
- Define `MASTER_ADDR` and `MASTER_PORT` as environment variables:
  ```
  export MASTER_ADDR=localhost
  export MASTER_PORT=12355
  ```

---

## Evaluation

To evaluate the model, the pipeline is similar to training. In the `Rainformer/evaluate` directory, you can run the evaluation script:
```
./submit_test.sh [job_name]
```
Which will submit the [`test.sh`](./Rainformer/evaluate/test.sh) script to SLURM. Then `test.sh` will run the [`test_on_cluster.py`](./Rainformer/evaluate/test_on_cluster.py) script with the specified configuration file (set in `test.sh`). An example configuration file is provided in [`evaluation_config.yaml`](./Rainformer/evaluate/test_configs/config.yaml).

### Evaluation on multiple lead times

To evaluate the model on a specific lead time, you can add the `--lead_time` argument to this line in the `test.sh` script:
```
srun --export=ALL python test_on_cluster.py --config_path=$CONFIG_FILE --lead_time=[lead time]
```
(or you can add it in the config file directly)

However, if you want to evaluate the model on multiple lead times, you can run the following command:
```
./submit_test_array.sh [job_name]
```
which works similarly to `submit_test.sh`, but it submits the [`test_array.sh`](./Rainformer/evaluate/test_array.sh) script to SLURM. This script will run the [`test_on_cluster.py`](./Rainformer/evaluate/test_on_cluster.py) script for each lead time specified in the `test_array.sh` file. Each variation of the evaluation script will be submitted as a separate job to SLURM and run in parallel if resources are available.

Results will be saved in the `results` directory.

## Citation

If you use Rainformer in your research, please cite the original paper:
```
@ARTICLE{9743916,
  author={Bai, Cong and Sun, Feng and Zhang, Jinglin and Song, Yi and Chen, Shengyong},
  journal={IEEE Geoscience and Remote Sensing Letters}, 
  title={Rainformer: Features Extraction Balanced Network for Radar-Based Precipitation Nowcasting}, 
  year={2022},
  volume={19},
  number={},
  pages={1-5},
  doi={10.1109/LGRS.2022.3162882}
}
```