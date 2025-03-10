import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import torch
from pysteps.visualization.precipfields import get_colormap

from test_on_cluster import setup_model, setup_data_module

import sys
sys.path.append('..')
from tool import undo_dgmr_prep, to_np


def visualize_single(sample_x, sample_y, prediction, mask, cmap_name='intensity', unit='mm/h', fname='visualise_single.png'):
    """
    Visualize the last input image along with ground truth and predicted outputs for a single sequence.

    Parameters:
    - sample_x: numpy array of shape (t_x, h, w), the input sequence.
    - sample_y: numpy array of shape (t_y, h, w), the ground truth output sequence.
    - prediction: numpy array of shape (t_y, h, w), the predicted output sequence.
    - mask: numpy array of shape (h, w), a boolean mask to apply to images.
    - cmap_name: str, colormap name for visualization (default is 'intensity').
    - unit: str, the unit for the colormap (default is 'mm/h').

    Returns:
    None, displays the visualization plot.
    """
    cmap, norm, _, _ = get_colormap(cmap_name, unit, 'pysteps')

    # Last input image
    last_input = sample_x[-1, :, :]

    n_cols = 11

    # Calculate rows required for ground truth and predictions
    n_rows_per_group = (len(sample_y) + n_cols - 2) // (n_cols - 1)
    n_rows_total = n_rows_per_group * 2  # Alternate GT and Pred rows

    fig, axes = plt.subplots(n_rows_total, n_cols, figsize=(30, 4 * n_rows_total))

    for ax_row in axes:
        for ax in ax_row:
            ax.axis('off')

    # Plot the last input image for ground truth and predictions
    axes[0][0].imshow(np.nan_to_num(last_input), cmap=cmap, norm=norm)
    axes[0][0].set_title("T=0")

    axes[1][0].imshow(np.nan_to_num(last_input), cmap=cmap, norm=norm)
    axes[1][0].set_title("T=0")

    # Plot ground truth and predicted images alternating row by row
    for idx in range(len(sample_y)):
        img_gt = np.nan_to_num(sample_y[idx])
        img_gt[~mask] = np.nan

        img_pred = np.nan_to_num(prediction[idx])
        img_pred[~mask] = np.nan

        # Ground truth row
        gt_row = (idx // (n_cols - 1)) * 2
        gt_col = idx % (n_cols - 1) + 1  # Start plotting from column 1
        axes[gt_row][gt_col].imshow(img_gt, cmap=cmap, norm=norm)
        axes[gt_row][gt_col].set_title(f"GT (T={idx + 1} min)")

        # Prediction row
        pred_row = gt_row + 1
        pred_col = gt_col
        axes[pred_row][pred_col].imshow(img_pred, cmap=cmap, norm=norm)
        axes[pred_row][pred_col].set_title(f"Pred (T={idx + 1} min)")

    plt.tight_layout()
    plt.savefig(f"../results/figures/{fname}")
    plt.show()


if __name__ == "__main__":
    #TODO: load checkpoint of pretrained model here and run inference on a batch
    #TODO: see how similar the predictions are to our model's predictions
    #TODO: does it also give blurry predictions? If not, what is the difference?
    net = setup_model(x_seq_size=4, y_seq_size=20, window_size=8, device='cpu')

    checkpoint_path = '/vol/knmimo-nobackup/users/pkools/thesis-forecasting/Rainformer/results/model_checkpoints/rainformer_ddp_18_11/best-epoch37-loss23364.7173.ckpt'
    
    checkpoint = None
    if checkpoint_path:
        if os.path.exists(checkpoint_path):
            print(f"Loading checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
        else:
            raise FileNotFoundError(f"Checkpoint file not found at {checkpoint_path}")
    
    if checkpoint is not None:
        net.load_state_dict(checkpoint['model_state_dict'])

    datamodule = setup_data_module(
        test_IDs_fn='/vol/knmimo-nobackup/users/pkools/thesis-forecasting/ldcast/data/knmi_data/data_splits/ldcast_test_202324_avg001mm_4x20y.npy',
        data_path='/vol/knmimo-nobackup/restore/knmimo/thesis_pelle/data/preprocessed/rtcor_prep',
        batch_size=2,
        x_seq_size=4,
        y_seq_size=20
    )

    datamodule.setup(stage='test')
    test_loader = datamodule.test_dataloader()
    batch = next(iter(test_loader))
    (x, y) = batch
    while isinstance(x, list) or isinstance(x, tuple):
        x = x[0][0]
    x, y = x.squeeze(), y.squeeze()

    # Run inference
    net.eval()
    with torch.no_grad():
        y_hat = net(x)

    # Undo preprocessing for visualization
    with tf.device("/cpu:0"):
        x_np = to_np(undo_dgmr_prep(x, downscale256=True))
        y_np = to_np(undo_dgmr_prep(y, downscale256=True))
        y_hat_np = to_np(undo_dgmr_prep(y_hat, downscale256=True))

    # Visualize batch
    #TODO: add dir_rtcor_recent and prefix_rtcor to config
    dir_rtcor_recent = "/vol/knmimo-nobackup/users/pkools/thesis-forecasting/data/rtcor-recent"
    prefix_rtcor_recent = "RAD_NL25_RAC_RT_"
    path = os.path.join(dir_rtcor_recent, '2019/{}201901010000.h5'.format(prefix_rtcor_recent))
    with h5py.File(path, 'r') as f:
        rain = f['image1']['image_data'][:]
        mask = ~(rain == 65535)

    for i, (x_sample, y_sample, y_hat_sample) in enumerate(zip(x_np, y_np, y_hat_np)):
        visualize_single(x_sample, y_sample, y_hat_sample, mask=mask, fname=f'visualise_{i}.png')

