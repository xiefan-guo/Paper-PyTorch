import argparse
import os
import time
import datetime
import numpy as np

from model import *
from dataset import *

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import datasets
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torchvision.transforms as transforms
from torchvision.utils import save_image


parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--dataset_name", type=str, default="facades", help="name of the dataset")
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--lambda_pixel", type=float, default=100.0, help="Loss weight of L1 pixel-wise loss")
parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_height", type=int, default=256, help="size of image height")
parser.add_argument("--img_width", type=int, default=256, help="size of image width")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=500, help="interval between image samples")
parser.add_argument("--checkpoint_interval", type=int, default=-1, help="interval between model checkpoints")
opt = parser.parse_args()
print(opt)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

os.makedirs("images/%s" % opt.dataset_name, exist_ok=True)
os.makedirs("saved_models/%s" % opt.dataset_name, exist_ok=True)

# PatchGAN: Calculate output of image discriminator
patch_size = (1, opt.img_height // 2 ** 4, opt.img_width // 2 ** 4)

# Configure dataloaders
transforms_ = [
    transforms.Resize((opt.img_height, opt.img_width)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
]
dataloader = DataLoader(
    ImageDataset("../../data/%s/%s" % (opt.dataset_name, opt.dataset_name), transforms_=transforms_),
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=opt.n_cpu
)
val_dataloader = DataLoader(
    ImageDataset("../../data/%s/%s" % (opt.dataset_name, opt.dataset_name), transforms_=transforms_, mode="val"),
    batch_size=10,
    shuffle=True,
    num_workers=0
)

generator = GeneratorUNet().to(device)
discriminator = Discriminator().to(device)

# Loss function
criterion_GAN = nn.MSELoss()
criterion_pixelwise = nn.L1Loss()

# Optimizer
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

# pre-trained models
if opt.epoch != 0:
    generator.load_state_dict(torch.load("saved_models/%s/generator_%d.pth" % (opt.dataset_name, opt.epoch)))
    discriminator.load_state_dict(torch.load("saved_models/%s/discriminator_%d.pth" % (opt.dataset_name, opt.epoch)))
else:
    generator.apply(weight_init_normal)
    discriminator.apply(weight_init_normal)

def sample_images(batches_done):

    imgs = next(iter(val_dataloader))
    real_A = Variable(imgs['B']).to(device)
    real_B = Variable(imgs['A']).to(device)
    fake_B = generator(real_A)
    img_sample = torch.cat((real_A.data, fake_B.data, real_B.data), -2)
    save_image(img_sample, "images/%s/%s.png" % (opt.dataset_name, batches_done), nrow=5, normalize=True)


# ---------
# Training
# ---------
prev_time = time.time()
print(prev_time)

for epoch in range(opt.epoch, opt.n_epochs):
    for i, imgs in enumerate(dataloader):

        real_A = Variable(imgs['B']).to(device)
        real_B = Variable(imgs['A']).to(device)

        # Adversarial ground truths
        valid = Variable(torch.ones(real_A.size(0), *patch_size)).to(device)
        fake = Variable(torch.zeros(real_A.size(0), *patch_size)).to(device)

        # ---------------
        # Train generator
        # ---------------

        optimizer_G.zero_grad()

        fake_B = generator(real_A)
        predict_fake = discriminator(fake_B, real_A)
        loss_GAN = criterion_GAN(predict_fake, valid)
        loss_pixel = criterion_pixelwise(fake_B, real_B)

        loss_G = loss_GAN + opt.lambda_pixel * loss_pixel

        loss_G.backward()
        optimizer_G.step()

        # -------------------
        # Train discriminator
        # -------------------

        optimizer_D.zero_grad()

        predict_real = discriminator(real_B, real_A)
        loss_real = criterion_GAN(predict_real, valid)
        predict_fake = discriminator(fake_B.detach(), real_A)
        loss_fake = criterion_GAN(predict_fake, fake)

        loss_D = (loss_real + loss_fake) * 0.5

        loss_D.backward()
        optimizer_D.step()

        # ---
        # Log
        # ---
        batches_done = epoch * len(dataloader) + i
        batches_left = opt.n_epochs * len(dataloader) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, pixel: %f, adv: %f, batches_done: %d] ETA: %s" %
            (epoch, opt.n_epochs, i, len(dataloader), loss_D.item(), loss_G.item(), loss_pixel.item(), loss_GAN.item(), batches_done, time_left)
        )

        if batches_done % opt.sample_interval == 0:
            sample_images(batches_done)

    if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:

        torch.save(generator.state_dict(), "saved_models/%s/generator_%d.pth" % (opt.dataset_name, epoch))
        torch.save(discriminator.state_dict(), "saved_models/%s/discriminator_%d.pth" % (opt.dataset_name, epoch))
