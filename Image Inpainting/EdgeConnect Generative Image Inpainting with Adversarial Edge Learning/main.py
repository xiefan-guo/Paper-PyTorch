import os
import cv2
import argparse
import numpy as np
from shutil import copyfile

import torch

from src.config import Config
from src.edge_connect import EdgeConnect


def main(mode=None):
    # ----------------------------------
    # starts the model
    # mode(int): 1 train, 2 test, 3 eval
    # ----------------------------------
    config = load_config(mode)

    # cuda visible devices
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(_) for _ in config.GPU)

    # init device
    if torch.cuda.is_available():
        config.DEVICE = torch.device('cuda')
        torch.backends.cudnn.benchmark = True  # cudnn auto-tuner
    else:
        config.DEVICE = torch.device('cpu')

    # 






def load_config(mode=None):

    pass



x = ['1', '2', '3']
y = ','.join(str(e) for e in x)
print(y)