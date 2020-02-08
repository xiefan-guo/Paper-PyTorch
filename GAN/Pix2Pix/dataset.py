import glob
import os
import random
import numpy as np

from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image


class ImageDataset(Dataset):

    def __init__(self, root, transforms_=None, mode="train"):

        self.transform = transforms.Compose(transforms_)
        self.files = sorted(glob.glob(os.path.join(root, mode) + "/*.*"))

        if mode == "train":
            self.files.extend(sorted(glob.glob(os.path.join(root, "test") + "/*.*")))

    def __getitem__(self, item):

        img = Image.open(self.files[item % len(self.files)])
        w, h = img.size
        img_A, img_B = img.crop((0, 0, w / 2, h)), img.crop((w / 2, 0, w, h))
        img_A, img_B = self.transform(img_A), self.transform(img_B)

        return {'A': img_A, 'B': img_B}

    def __len__(self):
        return len(self.files)