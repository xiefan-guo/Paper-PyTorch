import argparse
import os
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image
from torchvision import datasets

import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable


os.makedirs("images", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--n_classes", type=int, default=10, help="number of classes for dataset")
parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=1000, help="interval between image samples")
opt = parser.parse_args()
print(opt)

# the size of image
img_shape = (opt.channels, opt.img_size, opt.img_size)
print(*img_shape)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# Configure data loader
os.makedirs("../../data/mnist", exist_ok=True)
dataloader = DataLoader(
    dataset=datasets.MNIST(
        root="../../data/mnist",
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.Resize(opt.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        )
    ),
    batch_size=opt.batch_size,
    shuffle=True
)

class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()

        self.label_embedding = nn.Embedding(opt.n_classes, opt.n_classes)

        def generator_block(in_channels, out_channels, bn=True):
            layers = [nn.Linear(in_channels, out_channels)]
            if bn:
                layers.append(nn.BatchNorm1d(out_channels, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *generator_block(opt.latent_dim + opt.n_classes, 128, bn=False),
            *generator_block(128, 256),
            *generator_block(256, 512),
            *generator_block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, noise, label):
        img = torch.cat((self.label_embedding(label), noise), -1)
        img = self.model(img)
        img = img.view(img.size(0), *img_shape)
        return img


class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()

        self.label_embedding = nn.Embedding(opt.n_classes, opt.n_classes)

        self.model = nn.Sequential(
            nn.Linear(opt.n_classes + int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1)
        )

    def forward(self, img, label):
        img = torch.cat((img.view(img.size(0), -1), self.label_embedding(label)), -1)
        validity = self.model(img)
        return validity



# Loss functions
adversarial_loss = torch.nn.MSELoss()

# Initialize generator and discriminator
generator = Generator().to(device)
discriminator = Discriminator().to(device)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))


# --------
# Training
# --------
for epoch in range(opt.n_epochs):
    for i, (imgs, _) in enumerate(dataloader):

        # Adversarial ground truths
        valid = Variable(torch.ones(imgs.size(0), 1)).to(device)
        fake = Variable(torch.zeros(imgs.size(0), 1)).to(device)

        # Configure input
        real_imgs = Variable(imgs).to(device)
        _ = Variable(torch.LongTensor(_)).to(device)

        # ------------------
        # Training generator
        # ------------------

        optimizer_G.zero_grad()
        # noise
        z = Variable(torch.randn(imgs.size(0), opt.latent_dim)).to(device)
        gen_labels = Variable(torch.randint(0, opt.n_classes, (imgs.size(0),))).to(device)

        # Generate a batch of images
        gen_imgs = generator(z, gen_labels)

        # Loss measures generator's ability to fool the discriminator
        g_loss = adversarial_loss(discriminator(gen_imgs, gen_labels), valid)

        g_loss.backward()
        optimizer_G.step()

        # ----------------------
        # Training discriminator
        # ----------------------

        optimizer_D.zero_grad()

        # Measure discriminator's ability to classify real from generated samples
        real_loss = adversarial_loss(discriminator(real_imgs, gen_labels), valid)
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach(), gen_labels), fake)
        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
        )

        batches_done = epoch * len(dataloader) + i
        if batches_done % opt.sample_interval == 0:
            save_image(gen_imgs.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)

