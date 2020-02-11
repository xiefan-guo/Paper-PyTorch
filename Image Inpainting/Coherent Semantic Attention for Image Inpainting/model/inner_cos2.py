import torch
import torch.nn as nn

from torch.autograd import Variable

from tool import *


class InnerCos(nn.Module):

    def __init__(self, criterion='MSE', strength=1, skip=0, infe=None):
        super(InnerCos, self).__init__()

        self.criterion = nn.MSELoss() if criterion =='MSE' else nn.L1Loss()
        self.strenght = strength
        self.inin = None
        self.target = None
        # define whether this layer is skipped.
        self.skip = skip
        self.infe = infe

    def set_target(self, targetIn):
        self.target = targetIn

    def get_target(self):
        return self.target

    def set_mask(self, mask_global, opt):
        mask = cal_feat_mask(mask_global, 3, opt.threshold)
        self.mask = mask.squeeze()
        if torch.cuda.is_available():
            self.mask = self.mask.float().cuda()
        self.mask = Variable(self.mask, requires_grad=False)

    def forward(self, in_data):
        if not self.skip:
            self.former = in_data.narrow(1, 0, 512)  # 索引(*dimension*, *start*, *length*)
            self.bs, self.c, h, w = self.former.size()
            self.former_in_mask = torch.mul(self.former, self.mask)

            self.loss = self.criterion(self.former_in_mask * self.strength, self.target)
            self.output = in_data
        else:
            self.loss = 0
            self.output = in_data
        return self.output

    def backward(self, retain_graph=True):
        if not self.skip:
            self.loss.backward(retain_graph=retain_graph)
        return self.loss

    def __repr__(self):
        # 实例化对象，自我介绍
        skip_str = 'True' if not self.skip else 'False'
        return self.__class__.__name__ + '(' \
               + 'skip: ' + skip_str \
               + ' ,strength: ' + str(self.strength) + ')'

