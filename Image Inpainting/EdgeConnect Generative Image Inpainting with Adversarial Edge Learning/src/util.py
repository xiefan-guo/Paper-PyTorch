import os
import sys
import time
import random
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image


def create_mask(width, height, mask_width, mask_height, x=None, y=None):

    mask = np.zeros((height, width))
    mask_x = x if x is not None else random.randint(0, width - mask_width)
    mask_y = y if y is not None else random.randint(0, height - mask_height)
    mask[mask_y:mask_y + mask_height, mask_x:mask_x + mask_width] = 1

    return mask


def create_dir(dir):

    if not os.path.exists(dir):
        os.makedirs(dir)


def stitch_image(inputs, *outputs, img_per_row=2):

    gap = 5
    columns = len(outputs) + 1

    width, height = inputs[0][:, :, 0].shape
    img = Image.new('RGB', (width * img_per_row * columns + gap * (img_per_row - 1), height * int(len(inputs) / img_per_row)))



