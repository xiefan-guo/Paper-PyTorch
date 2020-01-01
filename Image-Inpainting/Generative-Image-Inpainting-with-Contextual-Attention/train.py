import argparse

import torch

parser = argparse.ArgumentParser()
parser.add_argument("--n_optim", type=int, default=1500, help="optim-step of training")
parser.add_argument("--blending_steps", type=int, default=1500, help="blending_steps of training")
parser.add_argument("--batch_size", type=int, default=12, help="size of the batches")
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


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
