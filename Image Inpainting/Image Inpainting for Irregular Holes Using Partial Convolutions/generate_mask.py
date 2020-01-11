import argparse
import os
import random
import numpy as np
from PIL import Image

action = [[0, 1], [0, -1], [1, 0], [-1, 0]]

def random_walk(canvas, init_x, init_y, length):

    x, y = init_x, init_y
    img_size = canvas.shape[-1]
    for i in range(length):
        id = random.randint(0, len(action) - 1)
        x = np.clip(x + action[id][0], a_min=0, a_max=img_size - 1)
        y = np.clip(y + action[id][1], a_min=0, a_max=img_size - 1)

        canvas[x][y] = 0

    return canvas


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--img_size', type=int, default=512)
    parser.add_argument('--mask_number', type=int, default=100)
    parser.add_argument('--save_dir', type=str, default='mask')
    opt = parser.parse_args()

    if not os.path.exists(opt.save_dir):
        os.makedirs(opt.save_dir, exist_ok=True)

    for i in range(opt.mask_number):

        canvas = np.ones((opt.img_size, opt.img_size)).astype('int')
        x = random.randint(0, opt.img_size - 1)  # [l, r]
        y = random.randint(0, opt.img_size - 1)

        mask = random_walk(canvas, x, y, opt.img_size ** 2)

        print("save: ", i, np.sum(mask), " | ", opt.img_size ** 2)

        img = Image.fromarray(mask * 255).convert('1')  # Binary image, 1 for white, 0 for black
        img.save('{:s}/{:04}.jpg'.format(opt.save_dir, i))
