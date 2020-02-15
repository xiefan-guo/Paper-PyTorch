import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


def weights_init(init_type='gaussian'):

    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find(
                'Linear') == 0) and hasattr(m, 'weight'):
            if init_type == 'gaussian':
                nn.init.normal_(m.weight, 0.0, 0.02)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    return init_fun


class VGG16FeatureExtractor(nn.Module):

    def __init__(self):
        super(VGG16FeatureExtractor, self).__init__()

        vgg16 = torchvision.models.vgg16(pretrained=True)
        self.enc_1 = nn.Sequential(*vgg16.features[:5])
        self.enc_2 = nn.Sequential(*vgg16.features[5:10])
        self.enc_3 = nn.Sequential(*vgg16.features[10:17])

        # fix the encoder
        for i in range(3):
            for param in getattr(self, 'enc_{:d}'.format(i + 1)).parameters():
                param.requires_grad = False

    def forward(self, img):
        results = [img]
        for i in range(3):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]


class PartialConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(PartialConv, self).__init__()

        self.img_conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.mask_conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)

        self.img_conv.apply(weights_init('kaiming'))
        nn.init.constant_(self.mask_conv.weight, 1.0)

        # mask is not updated
        for param in self.mask_conv.parameters():
            param.requires_grad = False

    def forward(self, input, mask):
        # http://masc.cs.gmu.edu/wiki/partialconv
        # C(X) = W^T * X + b, C(0) = b, D(M) = 1 * M + 0 = sum(M)
        # W^T* (M .* X) / sum(M) + b = [C(M .* X) – C(0)] / D(M) + C(0)

        output = self.img_conv(input * mask)
        if self.img_conv.bias is not None:
            output_bias = self.img_conv.bias.view(1, -1, 1, 1).expand_as(
                output)
        else:
            output_bias = torch.zeros_like(output)

        with torch.no_grad():
            output_mask = self.mask_conv(mask)

        no_update_holes = output_mask == 0
        mask_sum = output_mask.masked_fill_(no_update_holes, 1.0)

        output_pre = (output - output_bias) / mask_sum + output_bias
        output = output_pre.masked_fill_(no_update_holes, 0.0)

        new_mask = torch.ones_like(output)
        new_mask = new_mask.masked_fill_(no_update_holes, 0.0)

        return output, new_mask


class PConvBNActiv(nn.Module):

    def __init__(self, in_channels, out_channels, bn=True, sample='none-3', activ='relu', bias=False):
        super(PConvBNActiv, self).__init__()

        if sample == 'down-7':
            self.conv = PartialConv(in_channels, out_channels, kernel_size=7, stride=2, padding=3, bias=bias)
        elif sample == 'down-5':
            self.conv = PartialConv(in_channels, out_channels, kernel_size=5, stride=2, padding=2, bias=bias)
        elif sample == 'down-3':
            self.conv = PartialConv(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=bias)
        else:
            self.conv = PartialConv(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

        if bn:
            self.bn = nn.BatchNorm2d(out_channels)

        if activ == 'relu':
            self.activation = nn.ReLU()
        elif activ == 'leaky':
            self.activation = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, img, mask):
        img, mask = self.conv(img, mask)
        if hasattr(self, 'bn'):
            img = self.bn(img)
        if hasattr(self, 'activation'):
            img = self.activation(img)

        return img, mask


class PConvUNet(nn.Module):

    def __init__(self, in_channels=3, upsampling_mode='nearest'):
        super(PConvUNet, self).__init__()

        self.upsampling_mode = upsampling_mode
        self.freeze_enc_bn = False

        self.enc_PConv1 = PConvBNActiv(in_channels, 64, bn=False, sample='down-7')
        self.enc_PConv2 = PConvBNActiv(64, 128, sample='down-5')
        self.enc_PConv3 = PConvBNActiv(128, 256, sample='down-5')
        self.enc_PConv4 = PConvBNActiv(256, 512, sample='down-3')
        self.enc_PConv5 = PConvBNActiv(512, 512, sample='down-3')
        self.enc_PConv6 = PConvBNActiv(512, 512, sample='down-3')
        self.enc_PConv7 = PConvBNActiv(512, 512, sample='down-3')
        self.enc_PConv8 = PConvBNActiv(512, 512, sample='down-3')

        self.dec_PConv9 = PConvBNActiv(512 + 512, 512, activ='leaky')
        self.dec_PConv10 = PConvBNActiv(512 + 512, 512, activ='leaky')
        self.dec_PConv11 = PConvBNActiv(512 + 512, 512, activ='leaky')
        self.dec_PConv12 = PConvBNActiv(512 + 512, 512, activ='leaky')
        self.dec_PConv13 = PConvBNActiv(512 + 256, 256, activ='leaky')
        self.dec_PConv14 = PConvBNActiv(256 + 128, 128, activ='leaky')
        self.dec_PConv15 = PConvBNActiv(128 + 64, 64, activ='leaky')
        self.dec_PConv16 = PConvBNActiv(64 + in_channels, in_channels, bn=False, activ=None, bias=True)

    def forward(self, img, mask):
        enc_img_dict = {}
        enc_mask_dict = {}

        enc_img_dict['img0'], enc_mask_dict['img0'] = img, mask

        key_pre = 'img0'
        for i in range(1, 8 + 1):
            key = 'img{:d}'.format(i)
            enc = 'enc_PConv{:d}'.format(i)

            enc_img_dict[key], enc_mask_dict[key] = getattr(self, enc)(
                enc_img_dict[key_pre], enc_mask_dict[key_pre]
            )
            key_pre = key


        img, mask = enc_img_dict['img8'], enc_mask_dict['img8']

        for i in range(9, 16 + 1):
            key = 'img{:d}'.format(16 - i)
            dec = 'dec_PConv{:d}'.format(i)

            img = F.interpolate(img, scale_factor=2, mode=self.upsampling_mode)
            mask = F.interpolate(mask, scale_factor=2, mode='nearest')
            img = torch.cat((img, enc_img_dict[key]), dim=1)
            mask = torch.cat((mask, enc_mask_dict[key]), dim=1)

            img, mask = getattr(self, dec)(
                img, mask
            )

        return img, mask

    def train(self, mode=True):
        # ---------------------------------------------------------
        # Override the default train() to freeze the BN parameters
        # ---------------------------------------------------------
        super(PConvUNet, self).train(mode)  # 把父类的train()执行
        if self.freeze_enc_bn:
            for name, module in self.named_modules():
                if isinstance(module, nn.BatchNorm2d) and 'enc' in name:
                    module.eval()  # 不启用 BatchNormalization


