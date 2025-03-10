import os
import re
import sys
import time

import numpy as np
import torch
from tqdm import tqdm

from fire import Fire
from omegaconf import OmegaConf
from dotenv import load_dotenv

# Update the sys path and import custom utilities
sys.path.append('..')
import dataloading as dl
from Rainformer import Net
from tool import *

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
    device=None,
    input_h_w=[256, 256]
):
    print(f"input_h_w: {input_h_w}, type: {type(input_h_w)}")

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
        input_h_w=input_h_w
    ).to(device)
    return model

def load_checkpoint(checkpoint_path, device):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found at {checkpoint_path}")

    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Check if the checkpoint contains the entire training state or just the model's state_dict
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    return state_dict

def setup_data_module(
    test_IDs_fn,
    data_path,
    batch_size=28,
    x_seq_size=9,
    y_seq_size=9,
    h_w=(256, 256),
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
        h_w=h_w,
        vae_setup=False
    )
    datamodule.setup(stage='test')
    return datamodule


def test_model(device, net, test_loader, evaluate_upsampled=False, evaluate_first_n=None, use_orig_data=False, undo_prep=True):
    if use_orig_data:
        evaluate_upsampled = False

    thresholds = [0.5, 2, 5, 10, 30]
    CSI_DOWN, HSS_DOWN = [[] for _ in thresholds], [[] for _ in thresholds]
    CSI_UP, HSS_UP = [[] for _ in thresholds], [[] for _ in thresholds]
    mse_down, mae_down =[], []
    mse_up, mae_up = [], []

    # Test loop
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            (x,y) = batch

            if not use_orig_data:
                while isinstance(x, list) or isinstance(x, tuple):
                    x = x[0][0]

            x = x.squeeze().to(device)
            y = y.squeeze().to(device)

            # Run inference
            y_hat = net(x)

            if evaluate_first_n is not None:
                y_hat = y_hat[:, :evaluate_first_n, :, :]
                y = y[:, :evaluate_first_n, :, :]

            # Undo preprocessing for metrics
            if use_orig_data or not undo_prep:
                y_hat_down, y_down = y_hat, y
            else:
                y_hat_down, y_down = undo_dgmr_prep(y_hat, downscale256=False), undo_dgmr_prep(y, downscale256=False)

            mse_down.extend(to_np(B_mse_torch(y_down, y_hat_down)))
            mae_down.extend(to_np(B_mae_torch(y_down, y_hat_down)))

            csi_down = csi_torch(y_down, y_hat_down)
            hss_down = hss_torch(y_down, y_hat_down)

            if evaluate_upsampled:
                y_hat_up, y_up = undo_dgmr_prep(y_hat, downscale256=True), undo_dgmr_prep(y, downscale256=True)

                mse_up.extend(to_np(B_mse_torch(y_up, y_hat_up)))
                mae_up.extend(to_np(B_mae_torch(y_up, y_hat_up)))

                csi_up = csi_torch(y_up, y_hat_up)
                hss_up = hss_torch(y_up, y_hat_up)


            for t, thresh in enumerate(thresholds):
                CSI_DOWN[t].extend(to_np(csi_down[t]))
                HSS_DOWN[t].extend(to_np(hss_down[t]))
                if evaluate_upsampled:
                    CSI_UP[t].extend(to_np(csi_up[t]))
                    HSS_UP[t].extend(to_np(hss_up[t]))

    # Aggregate results
    for i in range(len(thresholds)):
        CSI_DOWN[i] = np.mean(CSI_DOWN[i])
        HSS_DOWN[i] = np.mean(HSS_DOWN[i])

        if evaluate_upsampled:
            CSI_UP[i] = np.mean(CSI_UP[i])
            HSS_UP[i] = np.mean(HSS_UP[i])
        
    mse_down_mean, mae_down_mean = np.mean(mse_down), np.mean(mae_down)
    if evaluate_upsampled:
        mse_up_mean, mae_up_mean = np.mean(mse_up), np.mean(mae_up)

    print("\nTesting Results of Downscaled (256x256) Predictions:")
    print("CSI:")
    for i, thresh in enumerate(thresholds):
        print(f"  r >= {thresh}: {CSI_DOWN[i]}")
    print("HSS:")
    for i, thresh in enumerate(thresholds):
        print(f"  r >= {thresh}: {HSS_DOWN[i]}")
    print(f"MSE: {mse_down_mean}  MAE: {mae_down_mean}")

    if evaluate_upsampled:
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
    }

    if evaluate_upsampled:
        metrics.update({
            'csi_up': CSI_UP,
            'hss_up': HSS_UP,
            'mse_up': mse_up_mean,
            'mae_up': mae_up_mean
        })

    return metrics


def setup_and_test_from_config(config_path, lead_time=None):
    config = OmegaConf.load(config_path)

    if lead_time is not None:
        config.lead_time = lead_time

    print("Final Config:")
    print(OmegaConf.to_yaml(config))

    # Pass the config values to the original setup_and_test function
    setup_and_test(
        batch_size=config.batch_size,
        test_IDs_fn=config.test_IDs_fn,
        checkpoint_path=config.checkpoint_path,
        data_path=config.data_path,
        x_seq_size=config.x_seq_size,
        y_seq_size=config.y_seq_size,
        window_size=config.window_size,
        evaluate_upsampled=config.get("evaluate_upsampled", False),
        evaluate_first_n=config.lead_time,
        input_h_w=config.input_h_w,
        use_orig_data=config.use_orig_data,
        undo_prep=config.undo_prep,
    )


#TODO: adjust to make more similar to setup_and_train
def setup_and_test(
    batch_size=16,
    test_IDs_fn=None,
    checkpoint_path=None,
    data_path=None,
    x_seq_size=4,
    y_seq_size=20,
    window_size=8,
    evaluate_upsampled=False,
    evaluate_first_n=None,
    input_h_w=[256, 256],
    use_orig_data=False,
    undo_prep=True
):
    print(f"input_h_w: {input_h_w}, type: {type(input_h_w)}")

    # Verify that required paths are provided
    if not (test_IDs_fn and checkpoint_path and data_path):
        raise ValueError("test_IDs_fn, checkpoint_path, and data_path must all be specified.")
    
    # torch.cuda.set_device(0)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Setup model and load checkpoint
    net = setup_model(x_seq_size=x_seq_size, y_seq_size=y_seq_size, window_size=window_size, device=device, input_h_w=input_h_w)
    state_dict = load_checkpoint(checkpoint_path, device)
    net.load_state_dict(state_dict, strict=False)
    net.eval()

    if use_orig_data:
        root = os.path.join(data_path, 'test_only_input-length_9_img-ahead_9_rain-threshhold_50.h5')
        _, test_data = get_data(root=root, test_only=True)
        test_loader = dl.get_original_dataloader('../../test_seq.npy', test_data, batch_size)
    else:
        # Setup test data module
        datamodule = setup_data_module(
            test_IDs_fn=test_IDs_fn,
            data_path=data_path,
            batch_size=batch_size,
            x_seq_size=x_seq_size,
            y_seq_size=y_seq_size,
            h_w=input_h_w
        )
        test_loader = datamodule.test_dataloader()

    # Metrics setup
    metrics = test_model(device, net, test_loader, evaluate_upsampled=evaluate_upsampled, evaluate_first_n=evaluate_first_n, use_orig_data=use_orig_data, undo_prep=undo_prep)

    # filename suffixes for saving metrics
    heavy_light_suffix = "_HEAVY" if "HEAVY" in test_IDs_fn else "_LIGHT" if "LIGHT" in test_IDs_fn else ""
    orig_data_suffix = "_orig_data" if use_orig_data else ""
    upsampled_suffix = "_down_and_upsampled" if evaluate_upsampled else ""
    first_n_suffix = '_all_outputs' if evaluate_first_n is None else f'_first_{evaluate_first_n}_outputs'
    
    model_version = checkpoint_path.split('/')[-2]
    epoch_pattern = re.compile(r'epoch(\d+)')  # Matches 'epoch' followed by a number
    ckpt_epoch = ("_epoch" + m.group(1) if (m := epoch_pattern.search(os.path.basename(checkpoint_path))) else "")
    ckpt_suffix = f'_{model_version}{ckpt_epoch}'

    np.save(f'../results/metrics{orig_data_suffix}{ckpt_suffix}_{x_seq_size}x{y_seq_size}y{heavy_light_suffix}{upsampled_suffix}{first_n_suffix}.npy', metrics)


def setup_and_test_from_config(config_path, lead_time=None):
    # Load the static config
    config = OmegaConf.load(config_path)

    # Dynamically add/override lead_time if provided
    if lead_time is not None:
        config.lead_time = lead_time

    print("Final Config:")
    print(OmegaConf.to_yaml(config))

    # Pass the config values to the original setup_and_test function
    setup_and_test(
        batch_size=config.batch_size,
        test_IDs_fn=config.test_IDs_fn,
        checkpoint_path=config.checkpoint_path,
        data_path=config.data_path,
        x_seq_size=config.x_seq_size,
        y_seq_size=config.y_seq_size,
        window_size=config.window_size,
        evaluate_upsampled=config.get("evaluate_upsampled", False),
        evaluate_first_n=config.lead_time,
        input_h_w=config.input_h_w,
        use_orig_data=config.use_orig_data,
        undo_prep=config.undo_prep,
    )

if __name__ == "__main__":
    # Fire(setup_and_test)
    Fire(setup_and_test_from_config)
