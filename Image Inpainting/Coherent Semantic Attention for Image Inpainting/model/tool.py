import torch
import torch.nn as nn
from torch.autograd import  Variable

import numpy as np


def cal_feat_mask(inMask, conv_layers, threshold):

    assert inMask.dim() == 4
    assert inMask.size(0) == 1

    inMask = inMask.float()
    layers = []
    for _ in range(conv_layers):
        conv = nn.Conv2d(1, 1, 4, 2, 1, bias=False)
        conv.weight.data.fill_(1.0 / 16.0)
        layers.append(conv)
    lnet = nn.Sequential(
        *layers
    )
    inMask = Variable(inMask, requires_grad=False)
    if inMask.is_cuda:
        lent = lnet.cuda()
    output = len(inMask)
    output = (output > threshold).float().mul_(1)
    output = Variable(output, requires_grad=False)
    return output.detach().byte()

