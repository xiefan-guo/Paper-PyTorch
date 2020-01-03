import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.autograd import Variable

from train import device


def extract_patches(x, size=3, step=1):
    """
    padding1(16 x 128 x 64 x 64) => (16 x 128 x 64 x 64 x 3 x 3)
    """
    zero_padding = nn.ZeroPad2d(1)
    x = zero_padding(x)  # padding with 0
    all_patches = x.unfold(2, size, step).unfold(3, size, step)
    # -----------------------------------------------------------------------------
    # torch.Tensor.unfold(dim, size, step)
    # Returns a tensor which contains all slices of size size in the dimension dim.
    # Step between two slices is given by step.
    # -----------------------------------------------------------------------------
    return all_patches


def down_sample(x, size=None, scale_factor=None, mode="nearest"):
    """
     downscaling resolution of foreground inputs
     before convolution and upscaling attention map after propagation.
    """
    if size is None:
        size = (int(scale_factor * x.size(2)), int(scale_factor * x.size(3)))
    # create coordinates
    h = torch.arange(0, size[0]) / (size[0] - 1.) * 2 - 1
    w = torch.arange(0, size[1]) / (size[1] - 1.) * 2 - 1
    # create grid
    grid = torch.zeros(size[0], size[1], 2)
    grid[:, :, 0] = w.unsqueeze(0).repeat(size[0], 1)
    grid[:, :, 1] = h.unsqueeze(0).repeat(size[1], 1).transpose(0, 1)
    # --------------------------------------------------------------
    # torch.unsqueeze(): Add a dimension to the specified dimension
    # x = torch.transpose(x, dim1, dim2): exchange dim1 and dim2
    # --------------------------------------------------------------

    # expand to match batch size
    grid = grid.unsqueeze(0).repeat(x.size(0), 1, 1, 1)

    grid = Variable(grid).to(device)
    # sampling
    return F.grid_sample(x, grid, mode=mode)
    # ---------------------------------------------------
    # F.grid_sample: https://www.wandouip.com/t5i207670/
    # ---------------------------------------------------


def reduce_mean(x):

    for i in range(4):
        if i != 1:
            x = torch.mean(x, dim=i, keepdim=True)

    return x


def l2_norm(x):

    x = x ** 2
    for i in range(4):
        if i != 1:
            x = torch.sum(x, dim=i, keepdim=True)

    return torch.sqrt(x)


def flow_to_image(flow):
    """Transfer flow map to image.
    Part of code forked from flownet.
    """
    out = []
    maxu = -999.
    maxv = -999.
    minu = 999.
    minv = 999.
    maxrad = -1
    for i in range(flow.shape[0]):
        u = flow[i, :, :, 0]
        v = flow[i, :, :, 1]
        idxunknow = (abs(u) > 1e7) | (abs(v) > 1e7)
        u[idxunknow] = 0
        v[idxunknow] = 0
        maxu = max(maxu, np.max(u))
        minu = min(minu, np.min(u))
        maxv = max(maxv, np.max(v))
        minv = min(minv, np.min(v))
        rad = np.sqrt(u ** 2 + v ** 2)
        maxrad = max(maxrad, np.max(rad))
        u = u/(maxrad + np.finfo(float).eps)
        v = v/(maxrad + np.finfo(float).eps)
        img = compute_color(u, v)
        out.append(img)
    return np.float32(np.uint8(out))




def compute_color(u,v):
    h, w = u.shape
    img = np.zeros([h, w, 3])
    nanIdx = np.isnan(u) | np.isnan(v)
    u[nanIdx] = 0
    v[nanIdx] = 0
    # colorwheel = COLORWHEEL
    colorwheel = make_color_wheel()
    ncols = np.size(colorwheel, 0)
    rad = np.sqrt(u**2+v**2)
    a = np.arctan2(-v, -u) / np.pi
    fk = (a+1) / 2 * (ncols - 1) + 1
    k0 = np.floor(fk).astype(int)
    k1 = k0 + 1
    k1[k1 == ncols+1] = 1
    f = fk - k0
    for i in range(np.size(colorwheel,1)):
        tmp = colorwheel[:, i]
        col0 = tmp[k0-1] / 255
        col1 = tmp[k1-1] / 255
        col = (1-f) * col0 + f * col1
        idx = rad <= 1
        col[idx] = 1-rad[idx]*(1-col[idx])
        notidx = np.logical_not(idx)
        col[notidx] *= 0.75
        img[:, :, i] = np.uint8(np.floor(255 * col*(1-nanIdx)))
    return img


def make_color_wheel():
    RY, YG, GC, CB, BM, MR = (15, 6, 4, 11, 13, 6)
    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros([ncols, 3])
    col = 0
    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.transpose(np.floor(255*np.arange(0, RY) / RY))
    col += RY
    # YG
    colorwheel[col:col+YG, 0] = 255 - np.transpose(np.floor(255*np.arange(0, YG) / YG))
    colorwheel[col:col+YG, 1] = 255
    col += YG
    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.transpose(np.floor(255*np.arange(0, GC) / GC))
    col += GC
    # CB
    colorwheel[col:col+CB, 1] = 255 - np.transpose(np.floor(255*np.arange(0, CB) / CB))
    colorwheel[col:col+CB, 2] = 255
    col += CB
    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.transpose(np.floor(255*np.arange(0, BM) / BM))
    col += + BM
    # MR
    colorwheel[col:col+MR, 2] = 255 - np.transpose(np.floor(255 * np.arange(0, MR) / MR))
    colorwheel[col:col+MR, 0] = 255
    return colorwheel



