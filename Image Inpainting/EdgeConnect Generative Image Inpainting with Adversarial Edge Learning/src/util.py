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




