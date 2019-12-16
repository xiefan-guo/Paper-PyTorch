# ----------
# Inpainting
# ----------
import torch
import torch.nn as nn
import os
import argparse

from models import *
from datasets import *
from utils import *

import torchvision.transforms as transforms
from torchvision.utils import save_image
from PIL import Image

from torch.utils.data import DataLoader
from torch.autograd import Variable

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
parser.add_argument("--prior_weight", type=float, default=0.003, help="lambda of prior loss")
parser.add_argument("--n_size", type=int, default=7, help="window size to W")
opt = parser.parse_args()
print(opt)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# Dataloader
transforms_ = [
    transforms.Resize((opt.img_size, opt.img_size)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
]
train_dataloader = DataLoader(
    InpaintingImageDataset("../../data/%s" % opt.dataset_name, transforms_=transforms_),
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=0,
)

# Initialize generator and discriminator
saved_state = torch.load(opt.model_path)
generator = Generator(opt).to(device)
discriminator = Discriminator(opt).to(device)
generator.load_state_dict(saved_state["G"])
discriminator.load_state_dict(saved_state["D"])

# Loss functions
context_loss = ContextLoss()
prior_loss = PriorLoss()

# ---------
# Training
# ---------
for i, (imgs, masked_imgs, mask) in enumerate(train_dataloader):
    z_optimum = nn.Parameter(
        torch.FloatTensor(np.random.normal(0, 1, (imgs.size(0), opt.latent_dim))).to(device)
    )
    optimizer_inpaint = torch.optim.Adam([z_optimum], lr=opt.lr, betas=(opt.b1, opt.b2))

    imgs = Variable(imgs).to(device)
    masked_imgs = Variable(masked_imgs).to(device)


    for epoch in range(opt.n_epochs):

        optimizer_inpaint.zero_grad()

        gen_imgs = generator(z_optimum)
        d_gen_imgs = discriminator(gen_imgs)

        mask_weight = create_mask_weight(mask.numpy(), opt.n_size)
        mask_weight = Variable(mask_weight).to(device)

        c_loss = context_loss(gen_imgs, masked_imgs, mask_weight)
        p_loss = prior_loss(opt.prior_weight, d_gen_imgs)

        inpainting_loss = c_loss + p_loss

        inpainting_loss.backward()
        optimizer_inpaint.step()

        print(
            "[Epoch %d/%d] [Batch %d/%d] [context_loss loss: %f] [prior_loss loss: %f]"
            % (epoch, opt.n_epochs, i, len(train_dataloader), c_loss.item(), p_loss.item())
        )

