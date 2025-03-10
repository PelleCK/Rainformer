# imports
import torch
import imageio
import matplotlib.pyplot as plt
import os
import numpy as np
import tensorflow as tf
import h5py

from test_on_cluster import setup_model, setup_data_module, load_checkpoint
from tool import undo_dgmr_prep, to_np
from pysteps.visualization.precipfields import get_colormap

# start with main function here
def main():
    # define model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    x_seq_size = 4
    y_seq_size = 9
    window_size = 8
    input_h_w = [256, 256]

    net = setup_model(
        x_seq_size=x_seq_size, 
        y_seq_size=y_seq_size, 
        window_size=window_size, 
        device=device, 
        input_h_w=input_h_w)

    # load model checkpoint
    ckpt_path = "/vol/knmimo-nobackup/users/pkools/thesis-forecasting/Rainformer/results/model_checkpoints/rainformer_4x9y_03_12/epoch23-loss7600.0625.ckpt"
    state_dict = load_checkpoint(ckpt_path, device)
    net.load_state_dict(state_dict, strict=False)
    net.eval()

    # load test data
    test_ids_fn = "/vol/knmimo-nobackup/restore/knmimo/thesis_pelle/data/preprocessed/data_splits/samples_4x9y.npy"
    data_path = "/vol/knmimo-nobackup/restore/knmimo/thesis_pelle/data/preprocessed/rtcor_prep"
    batch_size = 10
    datamodule = setup_data_module(
            test_IDs_fn=test_ids_fn,
            data_path=data_path,
            batch_size=batch_size,
            x_seq_size=x_seq_size,
            y_seq_size=y_seq_size,
            h_w=input_h_w
        )
    test_loader = datamodule.test_dataloader()

    test_ids = np.load(test_ids_fn, allow_pickle=True)

    # get first batch of data
    batch = next(iter(test_loader))
    x, y = batch
    while isinstance(x, list) or isinstance(x, tuple):
        x = x[0][0]

    x = x.squeeze().to(device)
    y = y.squeeze().to(device)

    # generate predictions
    net.eval()
    with torch.no_grad():
        y_hat = net(x)

    y_np = to_np(undo_dgmr_prep(y, downscale256=True))
    y_hat_np = to_np(undo_dgmr_prep(y_hat, downscale256=True))

    # generate gifs for each sample
    output_dir = "../results/figures/gifs/"
    os.makedirs(output_dir, exist_ok=True)

    #TODO: add dir_rtcor_recent and prefix_rtcor to config
    dir_rtcor_recent = "/vol/knmimo-nobackup/users/pkools/thesis-forecasting/data/rtcor-recent"
    prefix_rtcor_recent = "RAD_NL25_RAC_RT_"
    path = os.path.join(dir_rtcor_recent, '2019/{}201901010000.h5'.format(prefix_rtcor_recent))
    with h5py.File(path, 'r') as f:
        rain = f['image1']['image_data'][:]
        mask = ~(rain == 65535)
    
    cmap, norm, _, _ = get_colormap('intensity', 'mm/h', 'pysteps')

    for i, (y_hat_seq, y_gt_seq) in enumerate(zip(y_hat_np, y_np)):
        output_ts1 = test_ids[i][1][0]
        pred_images = []
        for y_hat_img in y_hat_seq:
            fig, ax = plt.subplots(figsize=(700 / 100, 765 / 100), dpi=100)  # Ensure 1:1 pixel representation
            img = np.nan_to_num(y_hat_img)
            img[~mask] = np.nan
            ax.imshow(img, cmap=cmap, norm=norm, interpolation='nearest')  # 'nearest' prevents smoothing
            ax.set_xlim(0, 700)
            ax.set_ylim(765, 0)
            ax.axis('off')
            fig.canvas.draw()

            # Convert plot to image array
            image = np.frombuffer(fig.canvas.buffer_rgba(), dtype='uint8')
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (4,))
            pred_images.append(image)
            plt.close(fig)

        gif_path = os.path.join(output_dir, f"prediction_{output_ts1}.gif")
        imageio.mimsave(gif_path, pred_images, fps=1)

        gt_images = []
        for y_gt_img in y_gt_seq:
            fig, ax = plt.subplots(figsize=(700 / 100, 765 / 100), dpi=100)  # Match dimensions
            img = np.nan_to_num(y_gt_img)
            img[~mask] = np.nan
            ax.imshow(img, cmap=cmap, norm=norm, interpolation='nearest')
            ax.set_xlim(0, 700)
            ax.set_ylim(765, 0)
            ax.axis('off')
            fig.canvas.draw()

            # Convert plot to image array
            image = np.frombuffer(fig.canvas.buffer_rgba(), dtype='uint8')
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (4,))
            gt_images.append(image)
            plt.close(fig)

        gif_path = os.path.join(output_dir, f"gt_{output_ts1}.gif")
        imageio.mimsave(gif_path, gt_images, fps=1)


    
    # # display the first gif
    # gif_path = os.path.join(output_dir, "prediction_0.gif")
    # gif = imageio.mimread(gif_path)
    # plt.imshow(gif[0])
    # plt.axis('off')
    # plt.show()

if __name__ == "__main__":
    main()
