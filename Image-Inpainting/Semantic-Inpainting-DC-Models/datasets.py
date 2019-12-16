import glob
import os
import random

import torch

import numpy as np
from PIL import Image

from torch.utils.data import Dataset
import torchvision.transforms as transforms


class GanImageDataset(Dataset):
    def __init__(self, root, transforms_=None):
        self.transform = transforms.Compose(transforms_)
        self.files = sorted(glob.glob("%s/*.jpg" % root))

    def __getitem__(self, index):
        img = Image.open(self.files[index % len(self.files)])
        img = self.transform(img)

        return img

    def __len__(self):
        return len(self.files)


class InpaintingImageDataset(Dataset):
    def __init__(self, root, transforms_=None, img_size=64, mask_size=32):
        self.transform = transforms.Compose(transforms_)
        self.img_size = img_size
        self.mask_size = mask_size
        self.files = sorted(glob.glob("%s/*.jpg" % root))

    def apply_random_mask(self, img):
        """Randomly masks image"""
        y1, x1 = np.random.randint(0, self.img_size - self.mask_size, 2)  # Return two random numbers in [l, R]
        y2, x2 = self.mask_size + y1, self.mask_size + x1
        mask = torch.ones_like(img)
        mask[:, y1:y2, x1:x2] = 0
        masked_img = img.clone()
        masked_img[:, y1:y2, x1:x2] = 0

        return masked_img, mask

    def __getitem__(self, index):
        img = Image.open(self.files[index % len(self.files)])
        img = self.transform(img)
        masked_img, aux = self.apply_random_mask(img)

        return img, masked_img, aux

    def __len__(self):
        return len(self.files)






