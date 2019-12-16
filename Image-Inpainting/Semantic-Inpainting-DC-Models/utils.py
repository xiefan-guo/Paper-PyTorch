import torch
import torch.nn as nn

import numpy as np
from scipy.signal import convolve2d

class ContextLoss(nn.Module):

    def __init__(self):
        super(ContextLoss, self).__init__()
        self.l1_loss = nn.L1Loss()

    def forward(self, gen_imgs, masked_imgs, mask_weight):
        loss = self.l1_loss(mask_weight * gen_imgs, mask_weight * masked_imgs)

        return loss


class PriorLoss(nn.Module):

    def __init__(self):
        super(PriorLoss, self).__init__()

    def forward(self, lam, d_gen_imgs):
        ones = torch.ones_like(d_gen_imgs)
        loss = lam * torch.log(ones - d_gen_imgs)
        loss = torch.mean(loss)

        return loss


def create_mask_weight(mask, n_size):

    kernel = np.ones((n_size, n_size), dtype=float)
    kernel = kernel / np.sum(kernel)

    mask_weight = np.zeros(mask.shape, dtype=float)

    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            mask_weight[i, j] = convolve2d(mask[i, j],
                                            kernel,
                                            mode='same',
                                            boundary='symm' )
    mask_weight = mask * ( 1.0 - mask_weight )

    return torch.FloatTensor(mask_weight)