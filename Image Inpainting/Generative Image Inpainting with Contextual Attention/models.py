import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

from utils import *
from train import device

# conv 1-6
class DownModule(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(DownModule).__init__()

        def conv_block(in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, activation=nn.ELU()):
            layers = [
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation=dilation),
                activation
            ]
            return layers

        self.model = nn.Sequential(
            *conv_block(in_ch, out_ch, kernel_size=5),
            *conv_block(out_ch, out_ch * 2, stride=2),
            *conv_block(out_ch * 2, out_ch * 2),
            *conv_block(out_ch * 2, out_ch * 4, stride=2),
            *conv_block(out_ch * 4, out_ch * 4),
            *conv_block(out_ch * 4, out_ch * 4)
        )

    def forward(self, x):
        x = self.model(x)
        return x


# conv 7-10
class DilationModule(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(DilationModule, self).__init__()

        def dilated_conv_block(in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, activation=nn.ELU()):
            layers = [
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation=dilation),
                activation
            ]
            return layers

        self.model = nn.Sequential(
            *dilated_conv_block(in_ch, out_ch, padding=2, dilation=2),
            *dilated_conv_block(in_ch, out_ch, padding=4, dilation=4),
            *dilated_conv_block(in_ch, out_ch, padding=8, dilation=8),
            *dilated_conv_block(in_ch, out_ch, padding=16, dilation=16)
        )

    def forward(self, x):
        x = self.model(x)
        return x


# conv 11-17
class UpModule(nn.Module):

    def __init__(self, in_ch, out_ch, isRefine=False):
        super(UpModule, self).__init__()

        def conv_block(in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, activation=nn.ELU()):
            layers = [
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation=dilation),
                activation
            ]
            return layers

        curr_dim = in_ch
        if isRefine:
            self.model1 = nn.Sequential(
                *conv_block(curr_dim, curr_dim // 2)
            )
            curr_dim //= 2
        else:
            self.model1 = nn.Sequential(
                *conv_block(curr_dim, curr_dim)
            )

        self.model2 = nn.Sequential(
            *conv_block(curr_dim, curr_dim),
            nn.Upsample(scale_factor=2),
            *conv_block(curr_dim, curr_dim // 2),
            *conv_block(curr_dim // 2, curr_dim // 2),
            nn.Upsample(scale_factor=2),
            *conv_block(curr_dim // 2, curr_dim // 4),
            *conv_block(curr_dim // 4, curr_dim // 8),
            *conv_block(curr_dim // 8, out_ch, activation=0)
        )

    def forward(self, x):
        x = self.model1(x)
        x = self.model2(x)
        return torch.clamp(x, min=-1., max=1.)


class FlattenModule(nn.Module):

    def __init__(self, in_ch, out_ch, isLocal=True):
        super(FlattenModule, self).__init__()

        def conv_block(in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, activation=nn.ELU()):
            layers = [
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation=dilation),
                activation
            ]
            return layers

        self.model1 = nn.Sequential(
            *conv_block(in_ch, out_ch, kernel_size=5, stride=2, padding=2, activation=nn.LeakyReLU()),
            *conv_block(out_ch, out_ch * 2, kernel_size=5, stride=2, padding=2, activation=nn.LeakyReLU()),
            *conv_block(out_ch * 2, out_ch * 4, kernel_size=5, stride=2, padding=2, activation=nn.LeakyReLU())
        )
        if isLocal:
            self.model2 = nn.Sequential(
                *conv_block(out_ch * 4, out_ch * 8, kernel_size=5, stride=2, padding=2, activation=nn.LeakyReLU())
            )
        else:
            self.model2 = nn.Sequential(
                *conv_block(out_ch * 4, out_ch * 4, kernel_size=5, stride=2, padding=2, activation=nn.LeakyReLU())
            )

    def forward(self, x):
        x = self.model1(x)
        x = self.model2(x)
        return x.view(x.size(0), -1)


class ContextualAttentionModule(nn.Module):

    def __init__(self, in_ch, out_ch, rate=2):
        super(ContextualAttentionModule, self).__init__()

        def conv_block(in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, activation=nn.ELU()):
            layers = [
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation=dilation),
                activation
            ]
            return layers

        self.rate = rate
        self.zero_padding = nn.ZeroPad2d(1)
        self.up_sample = nn.Upsample(scale_factor=self.rate, mode='nearest')
        self.model = nn.Sequential(
            *conv_block(in_ch, out_ch),
            *conv_block(in_ch, out_ch)
        )

    def forward(self, foreground, background, mask=None, kernel_size=3, stride=1, fuse_k=3,
                softmax_scale=10, training=True, fuse=True):
        """
        :param foreground: Input feature to match.
        :param background: Input feature for match.
        :param mask: Input mask for background, indicating patches not available.
        :param kernel_size: Kernel size for contextual attention.
        :param stride: Stride for extracting patches from background.
        :param softmax_scale: Scaled softmax for attention.
        :param training: Indicating if current graph is training or inference.
        :param rate: Dilation for matching.
        """

        # get shapes
        raw_f_size = foreground.size()  # B * 128 * 64 * 64
        raw_f_list_size = list(foreground.size())
        raw_b_list_size = list(background.size())
        print("list f b size:", raw_f_list_size, raw_b_list_size)

        # extract patches from background with stride and rate
        extract_size = self.rate * 2
        extract_step = self.rate
        raw_patches = extract_patches(background, size=extract_size, step=extract_step)
        print("raw_patches size:", raw_patches.size())
        # B * HW * C * K * K (B, 32 * 32, 128, 4, 4)
        raw_patches = raw_patches.contiguous().view(raw_b_list_size[0], -1, raw_b_list_size[1], extract_size, extract_size)
        print("raw_patches size:", raw_patches.size())
        # ----------------------------------------
        # torch.contiguous()
        # Provide continuous conditions for view()
        # ----------------------------------------

        # downscaling resolution of foreground inputs
        # before convolution and upscaling attention map after propagation.
        foreground = down_sample(foreground, scale_factor=1. / self.rate, mode="nearest")
        background = down_sample(background, scale_factor=1. / self.rate, mode="nearest")

        f_size = foreground.size()  # B * 128 * 32 * 32
        f_list_size = list(foreground.size())
        print("f_list_size:", f_list_size)
        f_groups = torch.split(foreground, 1, dim=0)
        print("f_groups:",len(f_groups), type(f_groups[0]), f_groups[0].size())
        # -------------------------------------------------------------------
        # torch.split(): Split tensors by batch dimension; tuple is returned
        # -------------------------------------------------------------------

        b_size = background.size()
        b_list_size = list(background.size())
        patches = extract_patches(background)
        # B * HW * C * K * K (B, 32 * 32, 128, 3, 3)
        print("patches size:", patches.size())
        patches = patches.contiguous().view(b_list_size[0], -1, b_list_size[1], kernel_size, kernel_size)
        print("patches size:", patches.size())
        patches_groups = torch.split(patches, 1, dim=0)
        raw_patches_groups = torch.split(raw_patches, 1, dim=0)

        # process mask
        if mask is not None:
            mask = down_sample(mask, scale_factor=1. / self.rate, mode="nearest")
        else:
            mask = Variable(torch.zeros(1, 1, b_size[2], b_size[3])).to(device)

        print("mask size:", mask.size())

        patches_mask = extract_patches(mask)
        print("patches_mask size:", patches_mask.size())
        patches_mask = patches_mask.contiguous().view(1, 1, -1, kernel_size, kernel_size)  # B * C * HW * K * K
        print("patches_mask size:", patches_mask.size())

        patches_mask = patches_mask[0]  # (1, 32 * 32, 3, 3)
        patches_mask = reduce_mean(patches_mask)
        eq_patches_mask = patches_mask.eq(0.).float()  # (1, 32 * 32, 1, 1)

        y = []
        offsets = []
        k = fuse_k
        scale = softmax_scale
        fuse_weight = Variable(torch.eye(k).view(1, 1, k, k)).to(device)  # Return Unit matrix (1, 1, k, k)

        for i_f, i_patches, i_raw_patches in zip(f_groups, patches_groups, raw_patches_groups):
            """
            i_f: separated tensor along batch dimension of front; (B=1, C=128, H=32, W=32)
            i_patches: separated patch tensor along batch dimension of back; (B=1, O=32*32, I=128, KH=3, KW=3)
            i_raw_patches: separated tensor along batch dimension of back; (B=1, I=32*32, O=128, KH=4, KW=4)
            """

            # conv for compare
            i_patches = i_patches[0]
            NaN = Variable(torch.FloatTensor([1e-4])).to(device)
            # print("l2_norm(i_patches) size:", l2_norm(i_patches).size())
            i_patches_normed = i_patches / torch.max(l2_norm(i_patches), NaN)
            print("i_patches_normed size:", i_patches_normed.size())
            i_f_conv = F.conv2d(i_f, i_patches_normed, stride=1, padding=1)  # (B=1, C=32*32, H=32, W=32)
            print("i_f_conv size:", i_f_conv.size())
            # ------------------------------------------------------------------------------------------------
            # torch.nn.functional.conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1)
            # input: (minibatch x in_channels x iH x iW)
            # weight:  (out_channels, in_channels/groups, kH, kW)
            # -------------------------------------------------------------------------------------------------

            # conv implementation for fuse scores to encourage large patches
            if fuse:
                i_f_conv = i_f_conv.view(1, 1, f_list_size[2] * f_list_size[3], b_list_size[2] * b_list_size[3])  # (B=1, I=1, H=32*32, W=32*32)
                i_f_conv = F.conv2d(i_f_conv, fuse_weight, stride=1, padding=1)  # (B=1, C=1, H=32*32, W=32*32)

                i_f_conv = i_f_conv.contiguous().view(1, f_list_size[2], f_list_size[3], b_list_size[2], b_list_size[3])  # # (B=1, 32, 32, 32, 32)
                i_f_conv = i_f_conv.permute(0, 2, 1, 4, 3)
                i_f_conv = i_f_conv.contiguous().view(1, 1, f_list_size[2] * f_list_size[3], b_list_size[2] * b_list_size[3])

                i_f_conv = F.conv2d(i_f_conv, fuse_weight, stride=1, padding=1)
                i_f_conv = i_f_conv.contiguous().view(1, f_list_size[3], f_list_size[2], b_list_size[3], b_list_size[2])
                i_f_conv.permute(0, 2, 1, 4, 3)

            i_f_conv = i_f_conv.contiguous().view(1, b_list_size[2] * b_list_size[3], f_list_size[2], f_list_size[3])  # (B=1, C=32*32, H=32, W=32)

            # softmax to match
            i_f_conv = i_f_conv * eq_patches_mask  # # mm => (1, 32 * 32, 1, 1)
            print("--------------------------\n softmax to match 1 i_f_conv:", i_f_conv.size())
            i_f_conv = F.softmax(i_f_conv * scale, dim=1)
            i_f_conv = i_f_conv * eq_patches_mask  # mask
            print("--------------------------\n softmax to match 1 i_f_conv:", i_f_conv.size())

            _, offset = torch.max(i_f_conv, dim=1)  # argmax, index (1 * 32 * 32)

            division = torch.div(offset, f_list_size[3]).long()
            print("division type size", type(division), division.size())
            offset = torch.stack([division, torch.div(offset, f_list_size[3]) - division], dim=1)
            print("offset type size", type(offset), offset.size())

            # deconv for patch pasting
            # 3.1 paste center
            i_raw_patches_center = i_raw_patches[0]
            print("i_raw_patches_center size", i_raw_patches_center.size())
            i_f_conv = F.conv_transpose2d(i_f_conv, i_raw_patches_center, stride=self.rate, padding=1) / 4.  # (B=1, C=128, H=64, W=64)
            # https://www.ptorch.com/docs/1/functional
            y.append(i_f_conv)
            offsets.append(offset)

        y = torch.cat(y, dim=0)  # back to the mini-batch
        y.contiguous().view(raw_f_list_size)
        offsets = torch.cat(offsets, dim=0)
        offsets = offsets.view([b_list_size[0]] + [2] + b_list_size[2:])

        # case1: visualize optical flow: minus current position
        h_add = Variable(torch.arange(0, float(b_list_size[2]))).to(device).view([1, 1, b_list_size[2], 1])
        h_add = h_add.expand(b_list_size[0], 1, b_list_size[2], b_list_size[3])
        w_add = Variable(torch.arange(0, float(b_list_size[3]))).to(device).view([1, 1, 1, b_list_size[3]])
        w_add = w_add.expand(b_list_size[0], 1, b_list_size[2], b_list_size[3])

        offsets = offsets - torch.cat([h_add, w_add], dim=1).long()

        # to flow image
        flow = torch.from_numpy(flow_to_image(offsets.permute(0, 2, 3, 1).cpu().data.numpy()))

        flow = flow.permute(0, 3, 1, 2)

        # # case2: visualize which pixels are attended
        # flow = torch.from_numpy(highlight_flow((offsets * mask.int()).numpy()))
        if self.rate != 1:
            flow = self.up_sample(flow)
        y = Variable(y).to(device)
        return self.model(y), flow


net = ContextualAttentionModule(128, 128).to(device)
print(net)
x = Variable(torch.ones(4, 128, 64, 64)).to(device)
out, f = net(x, x)
print(out.size(), f.size())




# -----------
# Main Models
# -----------

class Generator(nn.Module):
    def __init__(self, first_dim=32):
        super(Generator, self).__init__()
        self.stage_1 = CoarseNet(5, first_dim)
        self.stage_2 = RefinementNet(5, first_dim)

    def forward(self, masked_img, mask):  # mask : 1 x 1 x H x W
        # border, maybe
        mask = mask.expand(masked_img.size(0), 1, masked_img.size(2), masked_img.size(3))
        ones = to_var(torch.ones(mask.size()))

        # stage1
        stage1_input = torch.cat([masked_img, ones, ones * mask], dim=1)
        stage1_output, resized_mask = self.stage_1(stage1_input, mask[0].unsqueeze(0))

        # stage2
        new_masked_img = stage1_output * mask + masked_img.clone() * (1. - mask)
        stage2_input = torch.cat([new_masked_img, ones, ones * mask], dim=1)
        stage2_output, offset_flow = self.stage_2(stage2_input, resized_mask[0].unsqueeze(0))

        return stage1_output, stage2_output, offset_flow


class CoarseNet(nn.Module):
    '''
    # input: B x 5 x W x H
    # after down: B x 128(32*4) x W/4 x H/4
    # after atrous: same with the output size of the down module
    # after up : same with the input size
    '''

    def __init__(self, in_ch, out_ch):
        super(CoarseNet, self).__init__()
        self.down = DownModule(in_ch, out_ch)
        self.atrous = DilationModule(out_ch * 4, out_ch * 4)
        self.up = UpModule(out_ch * 4, 3)

    def forward(self, x, mask):
        x = self.down(x)
        resized_mask = down_sample(mask, scale_factor=0.25, mode='nearest')
        x = self.atrous(x)
        x = self.up(x)
        return x, resized_mask


class RefinementNet(nn.Module):
    '''
    # input: B x 5 x W x H
    # after down: B x 128(32*4) x W/4 x H/4
    # after atrous: same with the output size of the down module
    # after up : same with the input size
    '''

    def __init__(self, in_ch, out_ch):
        super(RefinementNet, self).__init__()
        self.down_conv_branch = DownModule(in_ch, out_ch)
        self.down_attn_branch = DownModule(in_ch, out_ch, activation=nn.ReLU())
        self.atrous = DilationModule(out_ch * 4, out_ch * 4)
        self.CAttn = ContextualAttentionModule(out_ch * 4, out_ch * 4)
        self.up = UpModule(out_ch * 8, 3, isRefine=True)

    def forward(self, x, resized_mask):
        # conv branch
        conv_x = self.down_conv_branch(x)
        conv_x = self.atrous(conv_x)

        # attention branch
        attn_x = self.down_attn_branch(x)
        attn_x, offset_flow = self.CAttn(attn_x, attn_x, mask=resized_mask)  # attn_x => B x 128(32*4) x W/4 x H/4

        # concat two branches
        deconv_x = torch.cat([conv_x, attn_x], dim=1)  # deconv_x => B x 256 x W/4 x H/4
        x = self.up(deconv_x)

        return x, offset_flow


class Discriminator(nn.Module):
    def __init__(self, first_dim=64):
        super(Discriminator, self).__init__()
        self.global_discriminator = FlattenModule(3, first_dim, False)
        self.local_discriminator = FlattenModule(3, first_dim, True)

    def forward(self, global_x, local_x):
        global_y = self.global_discriminator(global_x)
        local_y = self.local_discriminator(local_x)
        return global_y, local_y  # B x 256*(256 or 512)



