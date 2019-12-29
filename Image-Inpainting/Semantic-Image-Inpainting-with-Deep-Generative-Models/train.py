# ---------------
# Training DCGAN
# ---------------
import torch
import torch.nn as nn

import argparse
import os

import torchvision.transforms as transforms
from torchvision.utils import save_image
from PIL import Image

from torch.utils.data import DataLoader
from torch.autograd import Variable

from models import *
from datasets import *

os.makedirs("images", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=1000, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=8, help="size of the batches")
parser.add_argument("--dataset_name", type=str, default="img_align_celeba", help="name of the dataset")
parser.add_argument("--model_path", type=str, default="./checkpoints/model.pth", help="the parameter of dcgan")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=4, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=64, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=1000, help="interval between image sampling")
parser.add_argument("--checkpoint_interval", type=int, default=200, help="checkpoint interval")
opt = parser.parse_args()
print(opt)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# --------------
# Dataset loader
# --------------
transforms_ = [
    transforms.Resize((opt.img_size, opt.img_size)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]

train_dataloader = DataLoader(
    GanImageDataset("../../data/%s" % opt.dataset_name, transforms_=transforms_),
    batch_size=opt.batch_size,
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
generator = Generator(opt).to(device)
discriminator = Discriminator(opt).to(device)

# Loss function
adversarial_loss = nn.BCELoss()

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

# Initialize weight
if os.path.isfile(opt.model_path):
    saved_state = torch.load(opt.model_path)
    epoch = saved_state["epoch"]
    generator.load_state_dict(saved_state["G"])
    discriminator.load_state_dict(saved_state["D"])
    optimizer_G.load_state_dict(saved_state["optimizer_G"])
    optimizer_D.load_state_dict(saved_state["optimizer_D"])
else:
    generator.apply(weight_init_normal)
    discriminator.apply(weight_init_normal)

print(len(train_dataloader))

# ---------
# Training
# ---------
for epoch in range(opt.n_epochs):
    for i, (imgs) in enumerate(train_dataloader):

        # Adversarial ground truths
        valid = Variable(torch.ones(imgs.size(0), 1)).to(device)
        fake = Variable(torch.zeros(imgs.size(0), 1)).to(device)

        # Configure input
        real_imgs = Variable(imgs).to(device)

        # ------------------
        # Training generator
        # ------------------

        optimizer_G.zero_grad()
        # noise
        z = Variable(torch.randn(imgs.size(0), opt.latent_dim)).to(device)
        # Generate a batch of images
        gen_imgs = generator(z)

        # Loss measures generator's ability to fool the discriminator
        g_loss = adversarial_loss(discriminator(gen_imgs), valid)

        g_loss.backward()
        optimizer_G.step()

        # ----------------------
        # Training discriminator
        # ----------------------

        optimizer_D.zero_grad()

        # Measure discriminator's ability to classify real from generated samples
        real_loss = adversarial_loss(discriminator(real_imgs), valid)
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2.0

        d_loss.backward()
        optimizer_D.step()

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, opt.n_epochs, i, len(train_dataloader), d_loss.item(), g_loss.item())
        )

        batches_done = epoch * len(train_dataloader) + i
        if batches_done % opt.sample_interval == 0:
            save_image(gen_imgs.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)

    torch.save(
        {
            "epoch": epoch,
            "G": generator.state_dict(),
            "D": discriminator.state_dict(),
            "optimizer_G": optimizer_G.state_dict(),
            "optimizer_D": optimizer_D.state_dict()
        },
        opt.model_path
    )

    if epoch % opt.checkpoint_interval == 0:
        torch.save(
            {
                "epoch": epoch,
                "G": generator.state_dict(),
                "D": discriminator.state_dict(),
                "optimizer_G": optimizer_G.state_dict(),
                 "optimizer_D": optimizer_D.state_dict()
            },
            r"./checkpoints/{epoch}.pth".format(epoch=epoch)
        )
