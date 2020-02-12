import random

from PIL import Image
from glob import glob
from torch.utils.data import Dataset


class Place2(Dataset):

    def __init__(self, img_root, mask_root, img_transform, mask_transform, mode='train'):
        super(Place2, self).__init__()

        self.img_transform = img_transform
        self.mask_transform = mask_transform

        # if mode == 'train':
        #     self.files = glob('%s/train/*.*' % img_root)
        # else:
        #     self.files = glob('%s/%s/*.*' % (img_root, mode))

        self.files = glob('%s/*.*' % img_root)

        self.mask_files = glob('%s/*.*' % mask_root)
        self.n_mask = len(self.mask_files)

    def __getitem__(self, item):

        gt_img = Image.open(self.files[item % len(self.files)])
        gt_img = self.img_transform(gt_img.convert('RGB'))

        mask = Image.open(self.mask_files[random.randint(0, self.n_mask - 1)])
        mask = self.mask_transform(mask.convert('RGB'))

        return gt_img * mask, mask, gt_img

    def __len__(self):
        return len(self.files)

