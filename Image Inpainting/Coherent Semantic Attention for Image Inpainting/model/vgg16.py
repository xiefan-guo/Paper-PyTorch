import torch
import torch.nn as nn
import torchvision


class VGG16FeatureExtractor(nn.Module):

    def __init__(self, requires_grad=False):
        super(VGG16FeatureExtractor, self).__init__()

        vgg16 = torchvision.models.vgg16(pretrained=True)
        self.enc_1 = nn.Sequential(*vgg16.features[:5])
        self.enc_2 = nn.Sequential(*vgg16.features[5:10])
        self.enc_3 = nn.Sequential(*vgg16.features[10:17])
        self.enc_4 = nn.Sequential(*vgg16.features[17:23])

        # fix the encoder
        if not requires_grad:
            for i in range(4):
                for param in getattr(self, 'enc_{:d}'.format(i + 1)).parameters():
                    param.requires_grad = False

    def forward(self, img):
        results = [img]
        for i in range(4):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]
