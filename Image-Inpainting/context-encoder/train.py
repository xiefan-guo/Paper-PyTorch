import argparse
import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from models import *
from datasets import *

import torchvision.transforms as transforms
from torchvision.utils import save_image
from PIL import Image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

os.makedirs("images", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=20, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=8, help="size of the batches")
parser.add_argument("--dataset_name", type=str, default="img_align_celeba", help="name of the dataset")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=4, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=128, help="size of each image dimension")
parser.add_argument("--mask_size", type=int, default=64, help="size of random mask")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=769, help="interval between image sampling")
opt = parser.parse_args()
print(opt)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# Calculate output of image discriminator (PatchGAN)
patch_h, patch_w = int(opt.mask_size / 2 ** 3), int(opt.mask_size / 2 ** 3)
patch = (1, patch_h, patch_w)
print(patch)

# --------------
# Dataset loader
# --------------
transforms_ = [
    transforms.Resize((opt.img_size, opt.img_size), Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]
train_dataloader = DataLoader(
    ImageDataset("../../data/%s" % opt.dataset_name, transforms_=transforms_),
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=0,
)
test_dataloader = DataLoader(
    ImageDataset("../../data/%s" % opt.dataset_name, transforms_=transforms_, mode="val"),
    batch_size=12,
    shuffle=True,
    num_workers=0,
)

def weight_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


# Initialize generator and discriminator
generator = Generator(channels=opt.channels).to(device)
discriminator = Discriminator(channels=opt.channels).to(device)

# Initialize weight
generator.apply(weight_init_normal)
discriminator.apply(weight_init_normal)

# Loss functions
adversarial_loss = nn.BCELoss()
pixelwise_loss = nn.MSELoss()

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))


def save_sample(batches_done):
    samples, masked_samples, i = next(iter(test_dataloader))
    samples = Variable(samples).to(device)
    masked_samples = Variable(masked_samples).to(device)
    i = i[0].item()
    # Generate inpainted image
    gen_mask = generator(masked_samples)
    filled_samples = masked_samples.clone()
    filled_samples[:, :, i: i + opt.mask_size, i: i + opt.mask_size] = gen_mask
    # Save sample
    sample = torch.cat((masked_samples.data, filled_samples.data, samples.data), 2)
    save_image(sample, "images/%d.png" % batches_done, nrow=6, normalize=True)


# ---------
# Training
# ---------
# Multithreading training should be put in main
for epoch in range(opt.n_epochs):
    for i, (imgs, masked_imgs, masked_parts) in enumerate(train_dataloader):

        # Adversarial ground truths
        valid = Variable(torch.ones(imgs.size(0), 1)).to(device)
        fake = Variable(torch.zeros(imgs.size(0), 1)).to(device)

        # Configcure input
        imgs = Variable(imgs).to(device)
        masked_imgs = Variable(masked_imgs).to(device)
        masked_parts = Variable(masked_parts).to(device)

        # ---------------
        # Train Generator
        # ---------------

        optimizer_G.zero_grad()

        # Generate a batch of images
        gen_parts = generator(masked_imgs)

        # Compute loss
        g_adv = adversarial_loss(discriminator(gen_parts), valid)
        g_pixel = pixelwise_loss(gen_parts, masked_parts)

        g_loss = 0.999 * g_pixel + 0.001 * g_adv

        g_loss.backward()
        optimizer_G.step()

        # -------------------
        # Train Distriminator
        # -------------------

        optimizer_D.zero_grad()

        real_loss = adversarial_loss(discriminator(masked_parts), valid)
        fake_loss = adversarial_loss(discriminator(gen_parts.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2.0

        d_loss.backward()
        optimizer_D.step()

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f adv: %f pixel: %f]"
            % (epoch, opt.n_epochs, i, len(train_dataloader), d_loss.item(), g_loss.item(), g_adv.item(), g_pixel.item())
        )

        # Generator sample
        batches_done = epoch * len(train_dataloader) + i
        if batches_done % opt.sample_interval == 0:
            save_sample(batches_done)


"""
Error Log:
ValueError: Expected more than 1 value per channel when training, got input size torch.Size([1, 4000, 1, 1])
这个是在使用BatchNorm时不能把batchsize设置为1，一个样本的话y = (x - mean(x)) / (std(x) + eps)的计算中，x==mean(x)导致输出为0，
注意这个情况是在feature map为1的情况时，才可能出现x==mean(x)。
"""