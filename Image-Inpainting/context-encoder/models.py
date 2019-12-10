import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self, channels=3):
        super(Generator, self).__init__()

        def conv(in_feat, out_feat, bn=True):
            layers = [nn.Conv2d(in_feat, out_feat, 4, 2, 1)]
            if bn:
                layers.append(nn.BatchNorm2d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2))
            return layers
