import torch
import torch.nn as nn

from util import gram_matrix


def total_variation_loss(img):
    # shift one pixel and get difference
    loss = torch.mean(torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1])) + \
           torch.mean(torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :]))

    return loss


class TotalLoss(nn.Module):

    def __init__(self, extractor):
        super(TotalLoss, self).__init__()

        self.l1 = nn.L1Loss()
        self.extractor = extractor

    def forward(self, img, mask, output, gt):

        loss_dict = {}
        comp = mask * gt + (1 - mask) * output

        loss_dict['hole'] = self.l1((1 - mask) * output, (1 - mask) * gt)
        loss_dict['valid'] = self.l1(mask * output, mask * gt)

        if output.size(1) == 3:
            feat_comp = self.extractor(comp)
            feat_output = self.extractor(output)
            feat_gt = self.extractor(gt)
        elif output.size(1) == 1:
            feat_comp = self.extractor(torch.cat([comp] * 3, dim=1))
            feat_output = self.extractor(torch.cat([output] * 3, dim=1))
            feat_gt = self.extractor(torch.cat([gt] * 3, dim=1))
        else:
            raise ValueError('only gray rgb')

        loss_dict['perceptual'] = 0.0
        for i in range(3):
            loss_dict['perceptual'] += self.l1(feat_output, feat_gt)
            loss_dict['perceptual'] += self.l1(feat_comp, feat_gt)

        loss_dict['style'] = 0.0
        for i in range(3):
            loss_dict['style'] += self.l1(gram_matrix(feat_output[i]),
                                          gram_matrix(feat_gt[i]))
            loss_dict['style'] += self.l1(gram_matrix(feat_comp[i]),
                                          gram_matrix(feat_gt[i]))

        loss_dict['tv'] = total_variation_loss(comp)

        return loss_dict
