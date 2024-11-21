import numpy as np
import matplotlib.pyplot as plt
import h5py
import torch
from PIL import Image
import os

import torch
from torchmetrics.regression import CriticalSuccessIndex
from collections import deque
import tensorflow as tf

def show(a):
    plt.imshow(a)
    plt.show()

def to_np(a):
    return a.cpu().detach().numpy()

def get_data():
    root = '../../../../data/sf/KNMI/KNMI.h5'
    f = h5py.File(root, mode='r')
    train = f['train']
    test = f['test']
    train_image = train['images']
    test_image = test['images']
    return train_image, test_image


def psi(a, scale=4):
    # a shape is [B, S, H, W]
    B, S, H, W = a.shape
    C = scale ** 2
    new_H = int(H // scale)
    new_W = int(W // scale)
    a = np.reshape(a, (B, S, new_H, scale, new_W, scale))
    a = np.transpose(a, (0, 1, 3, 5, 2, 4))
    a = np.reshape(a, (B, S, C, new_H, new_W))
    return a

def inverse(a, scale=4):
    B, S, C, new_H, new_W = a.shape
    H = int(new_H * scale)
    W = int(new_W * scale)
    a = np.reshape(a, (B, S, scale, scale, new_H, new_W))
    a = np.transpose(a, (0, 1, 4, 2, 5, 3))
    a = np.reshape(a, (B, S, H, W))
    return a

def get_mask(eta, shape, test=False):
    B, S, C, H, W = shape
    if test:
        return torch.zeros((B, int(S // 2), C, H, W))
    eta -= 0.00002
    if eta < 0:
        eta = 0
    mask = np.random.random_sample((B, int(S // 2), C, H, W))
    mask[mask < eta] = 0
    mask[mask > eta] = 1
    return eta, torch.tensor(mask, dtype=torch.float)

def data_2_rnn_mask(data, batch, batch_size, sequence, scale, eta, test=False):
    sequence = sequence[batch - batch_size:batch]
    result = []
    for i in sequence:
        tmp = data[i] * 4783 / 100 * 12
        result.append(torch.tensor(tmp, dtype=torch.float))
    result = torch.stack(result, dim=0)
    result = psi(result, scale=scale)

    B, S, C, H, W = result.shape

    if test:
        return result, torch.zeros((B, int(S // 2), C, H, W))
    eta -= 0.00002
    if eta < 0:
        eta = 0

    mask = np.random.random_sample((B, int(S // 2), C, H, W))
    mask[mask < eta] = 0
    mask[mask > eta] = 1

    return result, torch.tensor(mask, dtype=torch.float), eta


def data_2_rnn(data, batch, batch_size, sequence, scale):
    sequence = sequence[batch - batch_size:batch]
    result = []
    for i in sequence:
        tmp = data[i] * 4783 / 100 * 12
        result.append(torch.tensor(tmp, dtype=torch.float))
    result = torch.stack(result, dim=0)
    result = psi(result, scale=scale)
    return result

def data_2_cnn(data, batch, batch_size, sequence):
    sequence = sequence[batch - batch_size:batch]
    result = []
    for i in sequence:
        tmp = data[i] * 4783 / 100 * 12
        result.append(torch.tensor(tmp, dtype=torch.float))
    result = torch.stack(result, dim=0)
    x = result[:, :9]
    y = result[:, 9:]
    return x, y

def data_2_cnn2(data, batch, batch_size, sequence):
    sequence = sequence[batch - batch_size:batch]
    result = []
    for i in sequence:
        tmp = data[i] * 4783 / 100 * 12
        result.append(torch.tensor(tmp, dtype=torch.float))
    result = torch.stack(result, dim=0)
    return result

def inverse_cnn2(x, y):
    x = torch.unsqueeze(x, dim=1)
    y = torch.unsqueeze(y, dim=1)
    x = to_np(x)
    y = to_np(y)
    x = inverse(x, scale=3)
    y = inverse(y, scale=3)

    x2 = np.zeros((x.shape[0], 9, 288, 288))
    y2 = np.zeros((y.shape[0], 9, 288, 288))

    index = 0

    for i in range(0, 864, 288):
        for j in range(0, 864, 288):
            x2[:, index] = x[:, 0, i:i+x2.shape[2], j:j+x2.shape[2]]
            y2[:, index] = y[:, 0, i:i+x2.shape[2], j:j+x2.shape[2]]
            index += 1
    return x2, y2


def _draw_color(t, flag, color):
    r = t[:, :, 0]
    g = t[:, :, 1]
    b = t[:, :, 2]
    r[flag] = color[0]
    g[flag] = color[1]
    b[flag] = color[2]
    return t



def draw_color_single(y):
    t = np.ones((y.shape[0], y.shape[1], 3)) * 255
    tt1 = []
    index = 0.5
    for i in range(30):
        tt1.append(index)
        index += 1
    color = [[28, 230, 180], [39, 238, 164], [58, 245, 143], [74, 248, 128], [97, 252, 108],
             [121, 254, 89], [143, 255, 73], [159, 253, 63], [173, 251, 56], [190, 244, 52],
             [203, 237, 52], [215, 229, 53], [227, 219, 56], [238, 207, 58], [246, 195, 58],
             [251, 184, 56], [254, 168, 51], [254, 153, 44], [253, 138, 38], [249, 120, 30],
             [244, 103, 23], [239, 88, 17], [231, 73, 12], [221, 61, 8], [212, 51, 5],
             [202, 42, 4], [188, 32, 2], [172, 23, 1], [158, 16, 1], [142, 10, 1]]

    for i in range(30):
        rain = y >= tt1[i]
        _draw_color(t, rain, color[i])
    #
    # rain_1 = y >= 0.5
    # rain_2 = y >= 2
    # rain_3 = y >= 5
    # rain_4 = y >= 10
    # rain_5 = y >= 30
    # _draw_color(t, rain_1, [156, 247, 144])
    # _draw_color(t, rain_2, [55, 166, 0])
    # _draw_color(t, rain_3, [103, 180, 248])
    # _draw_color(t, rain_4, [0, 2, 254])
    # _draw_color(t, rain_5, [250, 3, 240])
    t = t.astype(np.uint8)
    return t

def fundFlag(a, n, m):
    flag_1 = np.uint8(a >= n)
    flag_2 = np.uint8(a < m)
    flag_3 = flag_1 + flag_2
    return flag_3 == 2

def B_mse(a, b):
    mask = np.zeros(a.shape)
    mask[a < 2] = 1
    mask[fundFlag(a, 2, 5)] = 2
    mask[fundFlag(a, 5, 10)] = 5
    mask[fundFlag(a, 10, 30)] = 10
    mask[a > 30] = 30
    # n = a.shape[0] * b.shape[0] # this assumes square images
    n = a.size
    mse = np.sum(mask * ((a - b) ** 2)) / n
    return mse

def B_mse_torch(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Compute the Balanced Mean Squared Error (B-MSE) between two tensors.

    Heavy precipitation has higher weight

    Args:
        a: Ground truth tensor of shape (batch_size, sequence_length, height, width).
        b: Prediction tensor of shape (batch_size, sequence_length, height, width).

    Returns:
        B-MSE loss value for each image pair in the batch, flattened to shape (batch_size * sequence_length).
    """
    
    mask = torch.zeros(a.shape).to(a.device)
    mask[a < 2] = 1
    mask[(a >= 2) & (a < 5)] = 2
    mask[(a >= 5) & (a < 10)] = 5
    mask[(a >= 10) & (a < 30)] = 10
    mask[a > 30] = 30

    # weighted MSE based on thresholds, heavy precipitation has higher weight
    b_mse = torch.mean(mask * ((a - b) ** 2), dim=(2, 3))
    # flatten to get tensor of shape (batch_size * sequence_length), single value per image pair
    return b_mse.flatten()

def B_mae(a, b):
    mask = np.zeros(a.shape)
    mask[a < 2] = 1
    mask[fundFlag(a, 2, 5)] = 2
    mask[fundFlag(a, 5, 10)] = 5
    mask[fundFlag(a, 10, 30)] = 10
    mask[a > 30] = 30
    # n = a.shape[0] * b.shape[0]
    n = a.size
    mae = np.sum(mask * np.abs(a - b)) / n
    return mae

def B_mae_torch(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Compute the Balanced Mean Absolute Error (B-MAE) between two tensors.

    Heavy precipitation has higher weight

    Args:
        a: Ground truth tensor of shape (batch_size, sequence_length, height, width).
        b: Prediction tensor of shape (batch_size, sequence_length, height, width).

    Returns:
        B-MAE loss value for each image pair in the batch, flattened to shape (batch_size * sequence_length).
    """

    mask = torch.zeros(a.shape).to(a.device)
    mask[a < 2] = 1
    mask[(a >= 2) & (a < 5)] = 2
    mask[(a >= 5) & (a < 10)] = 5
    mask[(a >= 10) & (a < 30)] = 10
    mask[a > 30] = 30
    
    # weighted MAE based on thresholds, heavy precipitation has higher weight
    mae = torch.mean(mask * torch.abs(a - b), dim=(2, 3))
    # flatten to get tensor of shape (batch_size * sequence_length), single value per image pair
    return mae.flatten()

def draw_color(data):
    B, C, H, W = data.shape
    result = torch.zoers((B, C, H, W, 3))
    for i in range(B):
        for j in range(C):
            result[B, C] = draw_color_single(data[B, C])
    return result

def tp(pre, gt):
    return np.sum(pre * gt)

def fn(pre, gt):
    a = pre + gt
    flag = (gt == 1) & (a == 1)
    return np.sum(flag)

def fp(pre, gt):
    a = pre + gt
    flag = (pre == 1) & (a == 1)
    return np.sum(flag)

def tn(pre, gt):
    a = pre + gt
    flag = a == 0
    return np.sum(flag)

def _csi(pre, gt):
    eps = 1e-9
    TP, FN, FP, TN = tp(pre, gt), fn(pre, gt), fp(pre, gt), tn(pre, gt)
    return TP / (TP + FN + FP + eps)


def _hss(pre, gt):
    eps = 1e-9
    TP, FN, FP, TN = tp(pre, gt), fn(pre, gt), fp(pre, gt), tn(pre, gt)
    a = TP * TN - FN * FP
    b = (TP + FN) * (FN + TN) + (TP + FP) * (FP + TN) + eps
    if a / b < 0:
        return 0
    return a / b

def csi(pred, gt):
    threshold = [0.5, 2, 5, 10, 30]
    result = []
    for i in threshold:
        a = np.zeros(pred.shape)
        b = np.zeros(gt.shape)
        a[pred >= i] = 1
        b[gt >= i] = 1
        result.append(_csi(a, b))
    return result


def csi_torch(pred_batch: torch.Tensor, gt_batch: torch.Tensor) -> list[torch.Tensor]:
    """Compute the Critical Success Index (CSI) for a batch of predictions and ground truth.	

    Args:
        pred_batch: Batch of predictions of shape (batch_size, sequence_length, height, width).
        gt_batch: Batch of ground truth of shape (batch_size, sequence_length, height, width).

    Returns:
        List of CSI values for each threshold in [0.5, 2, 5, 10, 30].
    """

    pred_flat = pred_batch.flatten(end_dim=1)
    gt_flat = gt_batch.flatten(end_dim=1)
    threshold = [0.5, 2, 5, 10, 30]
    result = []
    for t in threshold:
        csi = CriticalSuccessIndex(threshold=t, keep_sequence_dim=0)
        result.append(csi(pred_flat, gt_flat))
    return result

def hss(pred, gt):
    threshold = [0.5, 2, 5, 10, 30]
    result = []
    for i in threshold:
        a = np.zeros(pred.shape)
        b = np.zeros(gt.shape)
        a[pred >= i] = 1
        b[gt >= i] = 1
        result.append(_hss(a, b))
    return result

def hss_torch(pred_batch: torch.Tensor, gt_batch: torch.Tensor) -> list[torch.Tensor]:
    """PyTorch implementation of hss function above.
    
    Args:
        pred_batch: Batch of predictions of shape (batch_size, sequence_length, height, width).
        gt_batch: Batch of ground truth of shape (batch_size, sequence_length, height, width).
        
    Returns:
        List of HSS values for each threshold in [0.5, 2, 5, 10, 30].
    """

    pred_flat = pred_batch.flatten(end_dim=1)
    gt_flat = gt_batch.flatten(end_dim=1)
    
    thresholds = [0.5, 2, 5, 10, 30]
    results = []
    
    for t in thresholds:
        # Apply thresholding to binarize predictions and ground truth
        pred_bin = (pred_flat >= t).float()
        gt_bin = (gt_flat >= t).float()

        TP = torch.sum(pred_bin * gt_bin, dim=(1, 2))  # True positives
        FN = torch.sum(gt_bin * (1 - pred_bin), dim=(1, 2))  # False negatives
        FP = torch.sum(pred_bin * (1 - gt_bin), dim=(1, 2))  # False positives
        TN = torch.sum((1 - pred_bin) * (1 - gt_bin), dim=(1, 2))  # True negatives

        numerator = TP * TN - FN * FP
        denominator = (TP + FN) * (FN + TN) + (TP + FP) * (FP + TN) + 1e-9

        hss = numerator / denominator

        hss = torch.where(hss < 0, torch.zeros_like(hss), hss)

        results.append(hss)
    
    return results


import torch
from torch import nn

class BMAEloss(nn.Module):
    """Balanced Mean Absolute Error (B-MAE) loss function for precipitation rate prediction."""

    def __init__(self, norm_dbz_values: bool=False):
        """Initializes the B-MAE loss function.
        
        Args:
            norm_dbz_values: whether data is converted to dBZ and normalized to the range [0, 1]
                if True, the loss function will map the thresholds to the corresponding normalized dBZ values
        """
        super(BMAEloss, self).__init__()

        self.norm_dbz_values = norm_dbz_values
        if self.norm_dbz_values: 
            # create mapping from mm/h thresholds to normalized dBZ values
            self.map_to_dbz = {}
            thresholds = [2, 5, 10, 30]
            for t in thresholds:
                self.map_to_dbz[t] = minmax(torch.tensor(t), norm_method='minmax', convert_to_dbz=True, undo=False).item()

    def fundFlag(self, a, n, m):
        if self.norm_dbz_values:
            n = self.map_to_dbz[n]
            m = self.map_to_dbz[m]

        flag_1 = (a >= n).int()
        flag_2 = (a < m).int()
        flag_3 = flag_1 + flag_2
        return flag_3 == 2

    def forward(self, pred, y):
        mask = torch.zeros(y.shape).to(y.device)
        mask[y < 2] = 1
        mask[self.fundFlag(y, 2, 5)] = 2
        mask[self.fundFlag(y, 5, 10)] = 5
        mask[self.fundFlag(y, 10, 30)] = 10
        mask[y > 30] = 30
        return torch.sum(mask * torch.abs(y - pred))


# class BMAEloss_norm_dbz(nn.Module):
#     """Balanced Mean Absolute Error (B-MAE) loss function for precipitation rate prediction.
    
#     Adapted to work with values that are converted to dBZ and then normalized to the range [0, 1].

#     """
#     def __init__(self):
#         super(BMAEloss, self).__init__()
#         self.map_to_dbz = {}
#         thresholds = [2, 5, 10, 30]
#         for t in thresholds:
#             self.map_to_dbz[t] = minmax(t, norm_method='minmax', convert_to_dbz=True, undo=False)            

#     def fundFlag(self, a, n, m):
#         flag_1 = (a >= n).int()
#         flag_2 = (a < m).int()
#         flag_3 = flag_1 + flag_2
#         return flag_3 == 2

#     def forward(self, pred, y):
#         mask = torch.zeros(y.shape).to(y.device)
#         mask[y < 2] = 1
#         mask[self.fundFlag(y, 2, 5)] = 2
#         mask[self.fundFlag(y, 5, 10)] = 5
#         mask[self.fundFlag(y, 10, 30)] = 10
#         mask[y > 30] = 30
#         return torch.sum(mask * torch.abs(y - pred))
   

def r_to_dbz(r):
    '''
    Convert mm/h to dbz
    '''
    # Convert to dBZ
    return 10 * torch.log10(200*r**(8/5)+1) 

def dbz_to_r(dbz):
    '''
    Convert dbz to mm/h
    '''
    ratio = ((10**(dbz/10)-1)/200)
    ratio = torch.clamp(ratio, min=0)
    r = ratio**(5/8)
    return r

def minmax(x, norm_method='minmax', convert_to_dbz = False, undo = False):
    '''
    Performs minmax scaling to scale the images to range of 0 to 1.
    norm_method: 'minmax' or 'minmax_tanh'. If tanh is used than scale to -1 to 1 as tanh
                is used for activation function generator, else scale values to be between 0 and 1
    '''
    assert norm_method == 'minmax' or norm_method == 'minmax_tanh'
    
    # define max intensity as 100mm
    MIN = 0
    MAX = 100
    
    if not undo:
        if convert_to_dbz:
            MAX = 55
            x = r_to_dbz(x)
        # Set values over 100mm/h to 100mm/h
        x = torch.clamp(x, MIN, MAX)
        if norm_method == 'minmax_tanh':
            x = (x - MIN - MAX/2)/(MAX/2 - MIN) 
        else:
            x = (x - MIN)/(MAX - MIN)
    else:
        if convert_to_dbz:
            MAX = 55
        if norm_method == 'minmax_tanh':
            x = x*(MAX/2 - MIN) + MIN + MAX/2
        else:
            x = x*(MAX - MIN) + MIN           
    return x


def undo_dgmr_prep(
        x: torch.Tensor, 
        norm_method: str='minmax', 
        r_to_dbz: bool=True, 
        downscale256: bool=True, 
        resize_method: tf.image.ResizeMethod = tf.image.ResizeMethod.BILINEAR
        ):
    """Reverse the preprocessing steps applied to the input data

    Reverses the preprocessing steps applied to the input data, such as normalization, conversion to dbz, and downscaling.
    Undoing the preprocessing is necessary to compute metrics that assume the input is in mm/h and unnormalized.
    
    Args:
        x: input tensor
        norm_method: normalization method used
        r_to_dbz: whether the input was converted to dbz
        downscale256: whether the input was downscaled to 256x256 (if True, upscales to 765x700)
        resize_method: method used for resizing the image (default: tf.image.ResizeMethod.BILINEAR)

    Returns:
        tensor with preprocessing undone
    """

    if norm_method:
        x = minmax(x, norm_method = norm_method, convert_to_dbz = r_to_dbz, undo = True)
    if r_to_dbz:
        x = dbz_to_r(x)
    if downscale256:
        # Upsample the image using bilinear interpolation
        tf.config.set_visible_devices([], 'GPU') # ensure that the image is upsampled on the CPU
        device = x.device

        x = to_np(x.permute(0, 2, 3, 1))
        x_tf =  tf.image.resize(x, (768, 768), method=resize_method)
        # Original shape was 765x700, crop prediction so that it fits this
        x_tf = x_tf[:, :-3,:-68, :]

        # Convert back to torch tensor and move to original device
        x_np = tf.transpose(x_tf, perm=[0, 3, 1, 2]).numpy()
        x = torch.from_numpy(x_np).to(device)
    
    return x

class CheckpointSaver:
    """Utility class to save model checkpoints during training and keep track of the best checkpoints based on validation loss."""

    def __init__(self, ddp: bool, model_dir: str, k_best: int=3):
        """Initializes the CheckpointSaver.	
        
        Args:
            ddp: whether DistributedDataParallel (DDP) is used
            model_dir: directory to save the checkpoints
            k_best: number of best checkpoints to keep
        """
        self.model_dir = model_dir
        self.k_best = k_best
        self.best_checkpoints = []
        self.ddp = ddp
        
    def save_checkpoint(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer, lr_scheduler: torch.optim.lr_scheduler._LRScheduler, epoch: int, val_loss: float):
        """Saves a model checkpoint and updates the best checkpoints list if necessary.

        Args:

            model: model to save
            optimizer: optimizer to save
            lr_scheduler: learning rate scheduler to save
            epoch: current epoch
            val_loss: validation loss of the model
        """

        if self.ddp:
            model = model.module
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'lr_scheduler_state_dict': lr_scheduler.state_dict(),
            'epoch': epoch,
            'val_loss': val_loss
        }
        
        # Save last checkpoint
        last_checkpoint_path = os.path.join(self.model_dir, 'last.ckpt')
        torch.save(checkpoint, last_checkpoint_path)
        
        # Save top-k checkpoints
        if len(self.best_checkpoints) < self.k_best or val_loss < self.best_checkpoints[-1][0]:
            checkpoint_path = os.path.join(self.model_dir, f'best-epoch{epoch}-loss{val_loss:.4f}.ckpt')
            torch.save(checkpoint, checkpoint_path)
            
            self.best_checkpoints.append((val_loss, checkpoint_path))
            self.best_checkpoints = sorted(self.best_checkpoints, key=lambda x: x[0])
            
            # Remove worst checkpoint if we have more than k
            if len(self.best_checkpoints) > self.k_best:
                _, worst_checkpoint_path = self.best_checkpoints.pop()
                os.remove(worst_checkpoint_path)


# import matplotlib.pyplot as plt
# import numpy as np
#
# plt.figure(dpi=200)
#
# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.rcParams['axes.unicode_minus'] = False
#
# curve_type = 'mae'
#
# x = np.arange(5, 50, 5)
#
# conv_mae = np.load('ConvLSTM/' + curve_type + '.npy')
# rainformer_mae = np.load('Rainformer_2/' + curve_type + '.npy')
# pfst_mae = np.load('PFST/' + curve_type + '.npy')
# mim_mae = np.load('MIM/' + curve_type + '.npy')
# predrnn_mae = np.load('PredRNN/' + curve_type + '.npy')
# predrnnpp_mae = np.load('PredRNN++/' + curve_type + '.npy')
# causal_mae = np.load('CausalLSTM/' + curve_type + '.npy')
# sa_mae = np.load('SAConvLSTM/' + curve_type + '.npy')
#
# plt.xlabel('Prediction interval (min)')
# plt.ylabel(curve_type)
#
# line1, = plt.plot(x, conv_mae, label='ConvLSTM')
# line2, = plt.plot(x, predrnn_mae, label='PredRNN')
# line3, = plt.plot(x, predrnnpp_mae, label='PredRNN++')
# line4, = plt.plot(x, causal_mae, label='CausalLSTM')
# line5, = plt.plot(x, mim_mae, label='MIM')
# line6, = plt.plot(x, pfst_mae, label='PFST')
# line7, = plt.plot(x, sa_mae, label='SA-ConvLSTM')
# line8, = plt.plot(x, rainformer_mae, color='black', label='Rainformer')
#
# plt.legend(handles=[line1, line2, line3, line4, line5, line6, line7, line8], labels=['ConvLSTM', 'PredRNN', 'PredRNN++',
#                                                                               'CausalLSTM', 'MIM', 'PFST', 'SA-ConvLSTM', 'Rainformer'], loc='best')
# plt.show()
