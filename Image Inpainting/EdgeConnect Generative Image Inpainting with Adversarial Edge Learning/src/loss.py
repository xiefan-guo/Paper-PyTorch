import torch
import torch.nn as nn
import torchvision.models as models


class AdversarialLoss(nn.Module):
    # ---------------------------------------------------
    # Adversarial loss: https://arxiv.org/abs/1711.10337
    # ---------------------------------------------------
    def __init__(self, type='nsgan', target_real_label=1.0, target_fake_label=0.0):
        # ----------------------------
        # type = nsgan | lsgan | hinge
        # ----------------------------
        super(AdversarialLoss, self).__init__()

        self.type = type
        # -----------------------------------------------------------------------------------------------
        # pytorch中还有另外一种形式的参数：buffer，如果将参数保存到buffer中，则在optim.step的时候，它不会更新。
        # 可以通过网络(继承自nn.Module)自带的函数：self.register_buffer(‘variable_name’,var)来设置这种参数。
        # -----------------------------------------------------------------------------------------------
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))

        if type == 'nsgan':
            self.criterion = nn.BCELoss()
        elif type == 'lsgan':
            self.criterion = nn.MSELoss()
        elif type == 'hinge':
            self.criterion = nn.ReLU()

    def __call__(self, outputs, is_real, is_disc=None):

        if self.type == 'hinge':
            if is_disc:
                if is_real:
                    outputs = -outputs
                return self.criterion(1 + outputs).mean()
            else:
                return (-outputs).mean()
        else:
            labels = (self.real_label if is_real else self.fake_label).expand_as(outputs)
            loss = self.criterion(outputs, labels)
            return loss


class StyleLoss(nn.Module):
    # -------------------------------------------------------------
    # Perceptual loss, VGG-based
    # https://arxiv.org/abs/1603.08155
    # https://github.com/dxyang/StyleTransfer/blob/master/utils.py
    # -------------------------------------------------------------
    def __init__(self):
        super(StyleLoss, self).__init__()

        self.add_module('vgg', VGG19FeatureExtractor())
        self.criterion = nn.L1Loss()

    def gram_matrix(self, feat):

        (b, ch, h, w) = feat.size()
        feat = feat.view(b, ch, h * w)
        feat_t = feat.transpose(1, 2)
        gram = torch.bmm(feat, feat_t) / (ch * h * w)

        return gram

    def __call__(self, x, y):

        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        style_loss = 0.0
        style_loss += self.criterion(self.gram_matrix(x_vgg['relu2_2']), self.gram_matrix(y_vgg['relu2_2']))
        style_loss += self.criterion(self.gram_matrix(x_vgg['relu3_4']), self.gram_matrix(y_vgg['relu3_4']))
        style_loss += self.criterion(self.gram_matrix(x_vgg['relu4_4']), self.gram_matrix(y_vgg['relu4_4']))
        style_loss += self.criterion(self.gram_matrix(x_vgg['relu5_2']), self.gram_matrix(y_vgg['relu5_2']))

        return style_loss


class PerceptualLoss(nn.Module):
    # -------------------------------------------------------------
    # Perceptual loss, VGG-based
    # https://arxiv.org/abs/1603.08155
    # https://github.com/dxyang/StyleTransfer/blob/master/utils.py
    # -------------------------------------------------------------
    def __init__(self, weights = [1.0, 1.0, 1.0, 1.0, 1.0]):
        super(PerceptualLoss, self).__init__()

        self.add_module('vgg', VGG19FeatureExtractor())
        self.criterion = nn.L1Loss()
        self.weights = weights

    def __call__(self, x, y):

        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        perceptual_loss = 0.0
        perceptual_loss += self.weights[0] * self.criterion(x_vgg['relu1_1'], y_vgg['relu1_1'])
        perceptual_loss += self.weights[1] * self.criterion(x_vgg['relu2_1'], y_vgg['relu2_1'])
        perceptual_loss += self.weights[2] * self.criterion(x_vgg['relu3_1'], y_vgg['relu3_1'])
        perceptual_loss += self.weights[3] * self.criterion(x_vgg['relu4_1'], y_vgg['relu4_1'])
        perceptual_loss += self.weights[4] * self.criterion(x_vgg['relu5_1'], y_vgg['relu5_1'])

        return perceptual_loss


class VGG19FeatureExtractor(nn.Module):

    def __init__(self):
        super(VGG19FeatureExtractor, self).__init__()

        features = models.vgg19(pretrained=True).features
        self.relu1_1 = torch.nn.Sequential(*features[:2])
        self.relu1_2 = torch.nn.Sequential(*features[2:4])

        self.relu2_1 = torch.nn.Sequential(*features[4:7])
        self.relu2_2 = torch.nn.Sequential(*features[7:9])

        self.relu3_1 = torch.nn.Sequential(*features[9:12])
        self.relu3_2 = torch.nn.Sequential(*features[12:14])
        self.relu3_3 = torch.nn.Sequential(*features[14:16])
        self.relu3_4 = torch.nn.Sequential(*features[16:18])

        self.relu4_1 = torch.nn.Sequential(*features[18:21])
        self.relu4_2 = torch.nn.Sequential(*features[21:23])
        self.relu4_3 = torch.nn.Sequential(*features[23:25])
        self.relu4_4 = torch.nn.Sequential(*features[25:27])

        self.relu5_1 = torch.nn.Sequential(*features[27:30])
        self.relu5_2 = torch.nn.Sequential(*features[30:32])
        self.relu5_3 = torch.nn.Sequential(*features[32:34])
        self.relu5_4 = torch.nn.Sequential(*features[34:36])
        # ------------------------------------------------
        # don't need the gradients, just want the features
        # ------------------------------------------------
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        relu1_1 = self.relu1_1(x)
        relu1_2 = self.relu1_2(relu1_1)

        relu2_1 = self.relu2_1(relu1_2)
        relu2_2 = self.relu2_2(relu2_1)

        relu3_1 = self.relu3_1(relu2_2)
        relu3_2 = self.relu3_2(relu3_1)
        relu3_3 = self.relu3_3(relu3_2)
        relu3_4 = self.relu3_4(relu3_3)

        relu4_1 = self.relu4_1(relu3_4)
        relu4_2 = self.relu4_2(relu4_1)
        relu4_3 = self.relu4_3(relu4_2)
        relu4_4 = self.relu4_4(relu4_3)

        relu5_1 = self.relu5_1(relu4_4)
        relu5_2 = self.relu5_2(relu5_1)
        relu5_3 = self.relu5_3(relu5_2)
        relu5_4 = self.relu5_4(relu5_3)

        out = {
            'relu1_1': relu1_1,
            'relu1_2': relu1_2,

            'relu2_1': relu2_1,
            'relu2_2': relu2_2,

            'relu3_1': relu3_1,
            'relu3_2': relu3_2,
            'relu3_3': relu3_3,
            'relu3_4': relu3_4,

            'relu4_1': relu4_1,
            'relu4_2': relu4_2,
            'relu4_3': relu4_3,
            'relu4_4': relu4_4,

            'relu5_1': relu5_1,
            'relu5_2': relu5_2,
            'relu5_3': relu5_3,
            'relu5_4': relu5_4,
        }
        return out



