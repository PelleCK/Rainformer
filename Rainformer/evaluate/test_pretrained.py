import torch
import numpy as np
from Rainformer import Net
from tqdm import tqdm
import sys
import os
import time
from fire import Fire
import matplotlib.pyplot as plt

# Update the sys path and import custom utilities
sys.path.append('..')
from tool import *
import dataloading as dl
import tensorflow as tf

#TODO: merge git repos and update imports
#TODO: after merge, remove redundant functions from tool and use DGMR ones

def setup_model(
    x_seq_size=9,
    y_seq_size=9,
    hidden_dim=96,
    downscaling_factors=(4, 2, 2, 2),
    layers=(2, 2, 2, 2),
    heads=(3, 6, 12, 24),
    head_dim=32,
    window_size=9,
    relative_pos_embedding=True,
    device=None
):
    model = Net(
        input_channel=x_seq_size,
        output_channel=y_seq_size,
        hidden_dim=hidden_dim,
        downscaling_factors=downscaling_factors,
        layers=layers,
        heads=heads,
        head_dim=head_dim,
        window_size=window_size,
        relative_pos_embedding=relative_pos_embedding,
        input_h_w=[288, 288]
    ).to(device)
    return model


def setup_data_module(
    test_IDs_fn,
    data_path,
    batch_size=28,
    x_seq_size=9,
    y_seq_size=9
):
    # Load test IDs
    test_IDs = np.load(test_IDs_fn, allow_pickle=True)

    # Initialize KNMIDataModule for test setup
    datamodule = dl.KNMIDataModule(
        train_data=None,
        val_data=None,
        test_data=test_IDs,
        data_path=data_path,
        batch_size=batch_size,
        x_seq_size=x_seq_size,
        y_seq_size=y_seq_size,
        h_w=(288, 288),
        vae_setup=False
    )
    datamodule.setup(stage='test')
    return datamodule


def test_model(device, net, test_loader, evaluate_first_n=None):
    thresholds = [0.5, 2, 5, 10, 30]
    CSI_DOWN, CSI_UP, HSS_DOWN, HSS_UP = [[] for _ in thresholds], [[] for _ in thresholds], [[] for _ in thresholds], [[] for _ in thresholds]
    mse_down, mse_up, mae_down, mae_up = [], [], [], []

    # Test loop
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            (x,y) = batch
            while isinstance(x, list) or isinstance(x, tuple):
                x = x[0][0]
            x, y = x.squeeze().to(device), y.squeeze().to(device)

            # Run inference
            y_hat = net(x)

            if evaluate_first_n is not None:
                y_hat = y_hat[:, :evaluate_first_n, :, :]
                y = y[:, :evaluate_first_n, :, :]

            # Undo preprocessing for metrics
            y_hat_down, y_down = undo_dgmr_prep(y_hat, downscale256=False), undo_dgmr_prep(y, downscale256=False)
            y_hat_up, y_up = undo_dgmr_prep(y_hat, downscale256=True), undo_dgmr_prep(y, downscale256=True)

            mse_down.extend(to_np(B_mse_torch(y_down, y_hat_down)))
            mse_up.extend(to_np(B_mse_torch(y_up, y_hat_up)))
            mae_down.extend(to_np(B_mae_torch(y_down, y_hat_down)))
            mae_up.extend(to_np(B_mae_torch(y_up, y_hat_up)))

            csi_down, csi_up = csi_torch(y_down, y_hat_down), csi_torch(y_up, y_hat_up)
            hss_down, hss_up = hss_torch(y_down, y_hat_down), hss_torch(y_up, y_hat_up)

            for t, thresh in enumerate(thresholds):
                CSI_DOWN[t].extend(to_np(csi_down[t]))
                CSI_UP[t].extend(to_np(csi_up[t]))
                HSS_DOWN[t].extend(to_np(hss_down[t]))
                HSS_UP[t].extend(to_np(hss_up[t]))

            # Compute metrics
            # for i in range(y_down_np.shape[0]): # sequences in batch
            #     for j in range(y_down_np.shape[1]):  # images in sequence
            #         a_down, b_down = y_down_np[i, j], y_hat_down_np[i, j]
            #         a_up, b_up = y_up_np[i, j], y_hat_up_np[i, j]
            #         # print(f"a_down shape: {a_down.shape}, b_down shape: {b_down.shape}")
            #         # print(f"a_up shape: {a_up.shape}, b_up shape: {b_up.shape}")

            #         mse_down.append(B_mse(a_down, b_down))
            #         mae_down.append(B_mae(a_down, b_down))
            #         csi_result_down, hss_result_down = csi(a_down, b_down), hss(a_down, b_down)

            #         mse_up.append(B_mse(a_up, b_up))
            #         mae_up.append(B_mae(a_up, b_up))
            #         csi_result_up, hss_result_up = csi(a_up, b_up), hss(a_up, b_up)

            #         for t, thresh in enumerate(thresholds):
            #             CSI_DOWN[t].append(csi_result_down[t])
            #             HSS_DOWN[t].append(hss_result_down[t])

            #             CSI_UP[t].append(csi_result_up[t])
            #             HSS_UP[t].append(hss_result_up[t])


    # Aggregate results
    for i in range(len(thresholds)):
        CSI_DOWN[i] = np.mean(CSI_DOWN[i])
        HSS_DOWN[i] = np.mean(HSS_DOWN[i])

        CSI_UP[i] = np.mean(CSI_UP[i])
        HSS_UP[i] = np.mean(HSS_UP[i])

    mse_down_mean, mae_down_mean = np.mean(mse_down), np.mean(mae_down)
    mse_up_mean, mae_up_mean = np.mean(mse_up), np.mean(mae_up)

    print("\nTesting Results of Downscaled (256x256) Predictions:")
    print("CSI:")
    for i, thresh in enumerate(thresholds):
        print(f"  r >= {thresh}: {CSI_DOWN[i]}")
    print("HSS:")
    for i, thresh in enumerate(thresholds):
        print(f"  r >= {thresh}: {HSS_DOWN[i]}")
    print(f"MSE: {mse_down_mean}  MAE: {mae_down_mean}")

    print("\nTesting Results of Upscaled (765x700) Predictions:")
    print("CSI:")
    for i, thresh in enumerate(thresholds):
        print(f"  r >= {thresh}: {CSI_UP[i]}")
    print("HSS:")
    for i, thresh in enumerate(thresholds):
        print(f"  r >= {thresh}: {HSS_UP[i]}")
    print(f"MSE: {mse_up_mean}  MAE: {mae_up_mean}")

    metrics = {
        'csi_down': CSI_DOWN,
        'hss_down': HSS_DOWN,
        'mse_down': mse_down_mean,
        'mae_down': mae_down_mean,
        'csi_up': CSI_UP,
        'hss_up': HSS_UP,
        'mse_up': mse_up_mean,
        'mae_up': mae_up_mean
    }

    return metrics

#TODO: adjust to make more similar to setup_and_train
def setup_and_test(
    batch_size=16,
    test_IDs_fn=None,
    checkpoint_path=None,
    data_path=None,
    x_seq_size=9,
    y_seq_size=9,
    window_size=9,
    evaluate_first_n=None
):
    # Verify that required paths are provided
    if not (test_IDs_fn and checkpoint_path and data_path):
        raise ValueError("test_IDs_fn, checkpoint_path, and data_path must all be specified.")
    
    # torch.cuda.set_device(0)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Setup model and load checkpoint
    #TODO: use shared setup_model function
    net = setup_model(x_seq_size=x_seq_size, y_seq_size=y_seq_size, window_size=window_size, device=device)

    #TODO: use shared load_checkpoint function
    checkpoint = None
    if checkpoint_path:
        if os.path.exists(checkpoint_path):
            print(f"Loading checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device)
        else:
            raise FileNotFoundError(f"Checkpoint file not found at {checkpoint_path}")
    
    if checkpoint is not None:
        net.load_state_dict(checkpoint, strict=False)
    net.eval()

    # Setup test data module
    #TODO: use shared setup_data_module function
    datamodule = setup_data_module(
        test_IDs_fn=test_IDs_fn,
        data_path=data_path,
        batch_size=batch_size,
        x_seq_size=x_seq_size,
        y_seq_size=y_seq_size
    )
    test_loader = datamodule.test_dataloader()

    # Metrics setup
    metrics = test_model(device, net, test_loader, evaluate_first_n=evaluate_first_n)

    # Save metrics
    # metrics = {
    #     'csi': csi,
    #     'hss': hss,
    #     'mse': mse,
    #     'mae': mae
    # }
    first_n = '_all_steps' if evaluate_first_n is None else f'_first_{evaluate_first_n}_steps'
    np.save(f'../results/eval_metrics_{x_seq_size}x{y_seq_size}{first_n}.npy', metrics)


if __name__ == "__main__":
    Fire(setup_and_test)
