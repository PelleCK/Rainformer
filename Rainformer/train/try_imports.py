import os
import sys

from dotenv import load_dotenv
sys.path.append("../..")
sys.path.append("../")

import inspect
import random
import time
from typing import Tuple, Optional

import numpy as np
import torch
import torch.distributed as dist
from torch import optim
from torch.nn.parallel import DistributedDataParallel as DDP

import wandb
from fire import Fire
from omegaconf import OmegaConf
from tqdm import tqdm

import dataloading as dl
from Rainformer import Net
from tool import *

from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn import Module