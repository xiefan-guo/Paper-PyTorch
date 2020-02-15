import os
import argparse
import numpy as np

import torch
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from evaluation import evaluate
from loss import TotalLoss
from model import PConvUNet, VGG16FeatureExtractor
from dataset import Place2
from util import load_ckpt, save_ckpt

parser = argparse.ArgumentParser()
parser.add_argument('--root', type=str, default='place2')
parser.add_argument('--mask_root', type=str, default='mask')
parser.add_argument('--save_dir', type=str, default='snapshots/default')
parser.add_argument('--log_dir', type=str, default='logs/default')
parser.add_argument('--lr', type=float, default=0.0002)
parser.add_argument('--lr_finetune', type=float, default=0.00005)
parser.add_argument('--n_epochs', type=int, default=1000)  # 1000000
parser.add_argument('--batch_size', type=int, default=6)
parser.add_argument('--n_threads', type=int, default=6)
parser.add_argument('--save_model_interval', type=int, default=100)  # 50000
parser.add_argument('--vis_interval', type=int, default=20)  # 5000
parser.add_argument('--log_interval', type=int, default=200)
parser.add_argument('--img_size', type=int, default=256)
parser.add_argument('--resume', type=str)
parser.add_argument('--finetune', action='store_true')  # 默认false，加参数--finetune选项为true
opt = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

if not os.path.exists(opt.save_dir):
    os.makedirs('{:s}/images'.format(opt.save_dir))
    os.makedirs('{:s}/ckpt'.format(opt.save_dir))

if not os.path.exists(opt.log_dir):
    os.makedirs(opt.log_dir)

writer = SummaryWriter(log_dir=opt.log_dir)

img_tf = transforms.Compose([
    transforms.Resize(size=(opt.img_size, opt.img_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=config.MEAN, std=config.STD)
])
mask_tf = transforms.Compose([
    transforms.Resize(size=(opt.img_size, opt.img_size)),
    transforms.ToTensor()
])

dataset_train = Place2("../../data/%s" % opt.root, opt.mask_root, img_tf, mask_tf, 'train')
dataset_val = Place2("../../data/%s" % opt.root, opt.mask_root, img_tf, mask_tf, 'val')

iterator_train = DataLoader(
    dataset_train,
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=0
)

net = PConvUNet().to(device)

# opt.finetune = True
if opt.finetune:
    print('finetune')
    lr = opt.lr_finetune
    net.freeze_enc_bn = True
else:
    lr = opt.lr

optimizer = torch.optim.Adam(
    filter(lambda p:p.requires_grad, net.parameters()),
    lr=lr,
    betas=(0.5, 0.999)
)

criterion = TotalLoss(VGG16FeatureExtractor()).to(device)

start_iter = 0
if opt.resume:
    start_iter = load_ckpt(
        opt.resume, [('model', net)], [('optimizer', optimizer)])
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    print('Starting from iter ', start_iter)

# --------
# Training
# --------
for epoch in range(start_iter, opt.n_epochs):
    net.train()
    for i, (image, mask, gt) in enumerate(iterator_train):

        image, mask, gt = Variable(image).to(device), Variable(mask, requires_grad=False).to(device), Variable(gt).to(device)

        output, _ = net(image, mask)
        loss_dict = criterion(image, mask, output, gt)

        loss = 0.0
        for key, coef in config.LAMBDA_DICT.items():
            val = coef * loss_dict[key]
            loss += val
            if (epoch + 1) % opt.log_interval == 0:
                writer.add_scalar('loss_{:s}'.format(key), val.item(), i + 1)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(
            "[Epoch %d/%d] [Batch %d/%d] [valid loss: %f] [hole loss: %f] [perceptual loss: %f] [style loss: %f] "
            "[tv loss: %f] [total loss: %f]" %
            (epoch, opt.n_epochs, i, len(iterator_train), loss_dict['valid'], loss_dict['hole'], loss_dict['perceptual'],
             loss_dict['style'], loss_dict['tv'], loss)
        )

        if ((epoch + 1) % opt.save_model_interval == 0 or (epoch + 1) == opt.n_epochs) and i == len(iterator_train) - 1:
            save_ckpt('{:s}/ckpt/{:d}.pth'.format(opt.save_dir, epoch + 1),
                      [('model', net)], [('optimizer', optimizer)], epoch + 1)

        if ((epoch + 1) % opt.vis_interval == 0) and i == len(iterator_train) - 1:
            net.eval()
            evaluate(net, dataset_val, device,
                     '{:s}/images/test_{:d}.jpg'.format(opt.save_dir, epoch + 1))

writer.close()