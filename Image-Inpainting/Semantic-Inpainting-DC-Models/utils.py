import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from scipy.signal import convolve2d

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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


def image_gradient(img):
    a = torch.FloatTensor([[[[1, 0, -1],
                             [2, 0, -2],
                             [1, 0, -1]],
                            [[1, 0, -1],
                             [2, 0, -2],
                             [1, 0, -1]],
                            [[1, 0, -1],
                             [2, 0, -2],
                             [1, 0, -1]]
                            ]]).to(device)
    G_x = F.conv2d(img, a, padding=1)
    b = torch.FloatTensor([[[[1, 2, 1],
                             [0, 0, 0],
                             [-1, -2, -1]],
                            [[1, 2, 1],
                             [0, 0, 0],
                             [-1, -2, -1]],
                            [[1, 2, 1],
                             [0, 0, 0],
                             [-1, -2, -1]]
                            ]]).to(device)
    G_y = F.conv2d(img, b, padding=1)
    return G_x, G_y


def posisson_blending(mask, gen_imgs, masked_imgs, opt):

    x = mask * masked_imgs + (1 - mask) * gen_imgs

    x_optimum = nn.Parameter(
        torch.FloatTensor(x.detach().cpu().numpy()).to(device)
    )

    optimizer_inpaint = torch.optim.Adam([x_optimum], lr=opt.lr, betas=(opt.b1, opt.b2))
    gen_imgs_G_x, gen_imgs_G_y = image_gradient(gen_imgs)

    for epoch in range(opt.blending_steps):

        optimizer_inpaint.zero_grad()

        x_G_x, x_G_y = image_gradient(x_optimum)
        blending_loss = torch.sum((x_G_x - gen_imgs_G_x) ** 2 + (x_G_y - gen_imgs_G_y) ** 2)

        blending_loss.backward()

        optimizer_inpaint.step()

        print(
            "[Epoch %d/%d] [blending loss: %f] "
            % (epoch, opt.blending_steps,  blending_loss.item())
        )

    return x_optimum.detach()