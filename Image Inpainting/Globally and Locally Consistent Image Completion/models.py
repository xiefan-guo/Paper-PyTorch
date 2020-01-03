import torch
import torch.nn as nn


class CompletionNetwork(nn.Module):

    def __init__(self):
        super(CompletionNetwork, self).__init__()

        def conv_block(in_channels, out_channels, kernel_size, stride, padding):
            layers = [
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            ]
            return layers

        def dilated_conv_block(in_channels, out_channels, kernel_size, stride, padding, dilation):
            layers = [
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation=dilation),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            ]
            return layers

        def deconv_block(in_channels, out_channels, kernel_size, stride, padding):
            layers = [
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            ]
            return layers

        def output(in_channels, out_channels, kernel_size, stride, padding):
            layers = [
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
                nn.Sigmoid()
            ]
            return layers

        self.model = nn.Sequential(
            *conv_block(4, 64, 5, 1, 2),  # input size: (batch_size, 4, img_h, img_w)

            *conv_block(64, 128, 3, 2, 1),  # input size: (batch_size, 64, img_h, img_w)
            *conv_block(128, 128, 3, 1, 1),  # input size: (batch_size, 128, img_h//2, img_w//2)

            *conv_block(128, 256, 3, 2, 1),  # input size: (batch_size, 128, img_h//2, img_w//2)
            *conv_block(256, 256, 3, 1, 1),  # input size: (batch_size, 256, img_h//4, img_w//4)
            *conv_block(256, 256, 3, 1, 1),  # input size: (batch_size, 256, img_h//4, img_w//4)
            *dilated_conv_block(256, 256, 3, 1, 2, 2),  # input size: (batch_size, 256, img_h//4, img_w//4)
            *dilated_conv_block(256, 256, 3, 1, 4, 4),  # input size: (batch_size, 256, img_h//4, img_w//4)
            *dilated_conv_block(256, 256, 3, 1, 8, 8),  # input size: (batch_size, 256, img_h//4, img_w//4)
            *dilated_conv_block(256, 256, 3, 1, 16, 16),  # input size: (batch_size, 256, img_h//4, img_w//4)
            *conv_block(256, 256, 3, 1, 1),  # input size: (batch_size, 256, img_h//4, img_w//4)
            *conv_block(256, 256, 3, 1, 1),  # input size: (batch_size, 256, img_h//4, img_w//4)

            *deconv_block(256, 128, 4, 2, 1),  # input size: (batch_size, 256, img_h//4, img_w//4)
            *conv_block(128, 128, 3, 1, 1),  # input size: (batch_size, 128, img_h//2, img_w//2)

            *deconv_block(128, 64, 4, 2, 1),  # input size: (batch_size, 128, img_h//2, img_w//2)
            *conv_block(64, 32, 3, 1, 1),  # input size: (batch_size, 64, img_h, img_w)
            *output(32, 3, 3, 1, 1)  # input size: (batch_size, 32, img_h, img_w)

            # output size: (batch_size, 3, img_h, img_w)
        )

    def forward(self, imgs):
        imgs = self.model(imgs)
        return imgs


class LocalDiscriminator(nn.Module):

    def __init__(self, input_size):
        super(LocalDiscriminator, self).__init__()

        def conv_block(in_channels, out_channels, kernel_size, stride, padding):
            layers = [
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            ]
            return layers

        self.in_channels, self.img_h, self.img_w = input_size[0], input_size[1], input_size[2]
        self.conv = nn.Sequential(
            *conv_block(self.in_channels, 64, 5, 2, 2),  # input size: (batch_size, in_channels, img_h, img_w)
            *conv_block(64, 128, 5, 2, 2),  # input size: (batch_size, 64, img_h//2, img_w//2)
            *conv_block(128, 256, 5, 2, 2),  # input size: (batch_size, 128, img_h//4, img_w//4)
            *conv_block(256, 512, 5, 2, 2),  # input size: (batch_size, 256, img_h//8, img_w//8)
            *conv_block(512, 512, 5, 2, 2)  # input size: (batch_size, 512, img_h//16, img_w//16)
        )
        in_features = 512 * (self.img_h // 32) * (self.img_w // 32)
        # input size: (batch_size, 512, img_h//32, img_w//32)
        self.fc = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.ReLU()
        )
        # output size: (batch_size, 1024)

    def forward(self, imgs):
        imgs = self.conv(imgs)
        imgs = imgs.view(imgs.size(0), -1)
        validity = self.fc(imgs)

        return validity


class GlobalDiscriminator(nn.Module):

    def __init__(self, input_size, dataset="celeba"):
        super(GlobalDiscriminator, self).__init__()

        def conv_block(in_channels, out_channels, kernel_size, stride, padding):
            layers = [
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            ]
            return layers

        self.in_channels, self.img_h, self.img_w = input_size[0], input_size[1], input_size[2]
        self.dataset = dataset
        self.conv1 = nn.Sequential(
            *conv_block(self.in_channels, 64, 5, 2, 2),  # input size: (batch_size, 32, img_h, img_w)
            *conv_block(64, 128, 5, 2, 2),  # input size: (batch_size, 64, img_h//2, img_w//2)
            *conv_block(128, 256, 5, 2, 2),  # input size: (batch_size, 128, img_h//4, img_w//4)
            *conv_block(256, 512, 5, 2, 2),  # input size: (batch_size, 256, img_h//8, img_w//8)
            *conv_block(512, 512, 5, 2, 2)  # input size: (batch_size, 512, img_h//16, img_w//16)
        )
        # input size: (batch_size, 512, img_h//32, img_w//32)
        if dataset == "celeba":
            in_features = 512 * (self.img_h // 32) * (self.img_w // 32)
            self.fc = nn.Sequential(
                nn.Linear(in_features, 1024),
                nn.ReLU()
            )
        elif dataset == "places2":
            self.conv2 = nn.Sequential(
                *conv_block(512, 512, 5, 2, 2)
            )
            in_features = 512 * (self.img_h // 64) * (self.img_w // 64)
            self.fc = nn.Sequential(
                nn.Linear(in_features, 1024),
                nn.ReLU()
            )
        else:
            raise ValueError("Unsupported architecture for %s" % (dataset))

        # output size: (batch_size, 1024)

    def forward(self, imgs):
        imgs = self.conv1(imgs)
        if self.dataset == "celeba":
            imgs = imgs.view(imgs.size(0), -1)
            validity = self.fc(imgs)
        elif self.dataset == "places2":
            imgs = self.conv2(imgs)
            imgs = imgs.view(imgs.size(0), -1)
            validity = self.fc(imgs)

        return validity


class ContextDiscriminator(nn.Module):

    def __init__(self, local_img_size, global_img_size, dataset="celeba"):
        super(ContextDiscriminator, self).__init__()

        self.local_model = LocalDiscriminator(local_img_size)
        self.global_model = GlobalDiscriminator(global_img_size, dataset=dataset)

        self.fc = nn.Sequential(
            nn.Linear(2048, 1),
            nn.Sigmoid()
        )

        # output size: (batch_size, 1)

    def forward(self, imgs):
        local_imgs, global_imgs = imgs
        local_validity = self.local_model(local_imgs)
        global_validity = self.global_model(global_imgs)

        validity = torch.cat((local_validity, global_validity), -1)
        validity = self.fc(validity)

        return validity


net = CompletionNetwork()
print(net)