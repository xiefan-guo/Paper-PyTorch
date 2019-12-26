import argparse
import os
import numpy as np

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.utils import save_image
from torch.autograd import Variable


os.makedirs("images", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=20, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--lambda_gp", type=int, default=10, help="lambda")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iter")
parser.add_argument("--sample_interval", type=int, default=1000, help="interval between image samples")
opt = parser.parse_args()
print(opt)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# the size of image
img_shape = (opt.channels, opt.img_size, opt.img_size)
print(*img_shape)

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

# ------
# Models
# ------
class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(opt.latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *img_shape)
        return img


class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)
        return validity


# Initialize generator and discriminator
generator = Generator().to(device)
discriminator = Discriminator().to(device)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))


def compute_GP(D, real_samples, fake_samples):

    alpha = Variable(torch.FloatTensor(np.random.random((real_samples.size(0), 1, 1, 1)))).to(device)

    hat_samples = (((alpha * real_samples) + (1 - alpha) * fake_samples))
    d_hat_samples = D(hat_samples)
    fake = Variable(torch.ones(real_samples.size(0), 1, requires_grad=False)).to(device)
    gradients = torch.autograd.grad(
        outputs=d_hat_samples,
        inputs=hat_samples,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2)  #　按１维度求范数

    return gradient_penalty


batches_done = 0

# --------
# Training
# --------
for epoch in range(opt.n_epochs):
    for i, (imgs, _) in enumerate(dataloader):

        real_imgs = Variable(imgs).to(device)

        # ----------------------
        # Training Discriminator
        # ----------------------

        optimizer_D.zero_grad()
        # noise
        z = Variable(torch.randn(imgs.size(0), opt.latent_dim)).to(device)

        fake_imgs = generator(z)

        # Real images
        real_validity = discriminator(real_imgs)
        # Fake images
        fake_validity = discriminator(fake_imgs)

        gradient_penalty = compute_GP(discriminator, real_imgs, fake_imgs)
        loss_D = -torch.mean(real_validity) + torch.mean(fake_validity) + opt.lambda_gp * torch.mean(gradient_penalty)

        loss_D.backward()
        optimizer_D.step()

        # ------------------
        # Training Generator
        # ------------------
        if i % opt.n_critic == 0:

            optimizer_G.zero_grad()

            gen_imgs = generator(z)
            # Adversarial loss
            loss_G = -torch.mean(discriminator(gen_imgs))

            loss_G.backward()
            optimizer_G.step()

            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, opt.n_epochs, i % len(dataloader), len(dataloader), loss_D.item(), loss_G.item())
            )

        if batches_done % opt.sample_interval == 0:
            save_image(gen_imgs.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)
        batches_done += 1


