import os
import argparse
import numpy as np

import torch

from torchvision import transforms
from torch.utils import data

import config

parser = argparse.ArgumentParser()
parser.add_argument('--root', type=str, default='')
parser.add_argument('--mask_root', type=str, default='mask')
parser.add_argument('--save_dir', type=str, default='snapshots/default')
parser.add_argument('--log_dir', type=str, default='')
parser.add_argument('--lr', type=float, default=0.0002)
parser.add_argument('--lr_fine_tune', type=float, default=0.00005)
parser.add_argument('--n_iter', type=int, default=1000000)
parser.add_argument('--batch_size', type=int, default=6)
parser.add_argument('--n_threads', type=int, default=6)
parser.add_argument('--save_model_interval', type=int, default=50000)
parser.add_argument('--vis_interval', type=int, default=5000)
parser.add_argument('--log_interval', type=int, default=10)
parser.add_argument('--img_size', type=int, default=512)
parser.add_argument('--resume', type=str)
parser.add_argument('--finetune', action='store_true')
opt = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if not os.path.exists(opt.save_dir):
    os.makedirs('{:s}/images'.format(opt.save_dir))
    os.makedirs('{:s}/ckpt'.format(opt.save_dir))

img_tf = transforms.Compose([
    transforms.Resize(size=(opt.img_size, opt.img_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=config.MEAN, std=config.STD)
])
mask_tf = transforms.Compose([
    transforms.Resize(size=(opt.img_size, opt.img_size)),
    transforms.ToTensor()
])


