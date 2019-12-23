import argparse
import os
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision import datasets


parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=10, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.001, help="adam: learning rate")
parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=100, help="interval between image samples")
opt = parser.parse_args()
print(opt)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# Configure data loader
os.makedirs("../../data/mnist", exist_ok=True)
train_dataloader = DataLoader(
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
test_dataloader = DataLoader(
    dataset=datasets.MNIST(
        root="../../data/mnist",
        train=False,
        download=True,
        transform=transforms.Compose(
            [transforms.Resize(opt.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        )
    ),
    batch_size=opt.batch_size,
    shuffle=False
)

class LeNet(nn.Module):

    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(  # input_size=(1*28*28)
                in_channels=1,
                out_channels=6,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # input_size=(6*28*28)，output_size=(6*14*14)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(  # input_size=(6*14*14)，output_size=16*10*10
                in_channels=6,
                out_channels=16,
                kernel_size=5,
                stride=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # input_size=(16*10*10)，output_size=(16*5*5)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(120, 84),
            nn.ReLU(),
        )
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)  # 全连接层均使用的nn.Linear()线性结构，输入输出维度均为一维，故需要把数据拉为一维
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


lenet = LeNet().to(device)
loss_func = nn.CrossEntropyLoss()  # 多分类问题，选择交叉熵损失函数
optimizer = optim.SGD(lenet.parameters(), lr=opt.lr, momentum=0.9)  # Momentum 优化器

# --------
# Training
# --------
for epoch in range(opt.n_epochs):
    for i, (imgs, _) in enumerate(train_dataloader):
        imgs, _ = Variable(imgs).to(device), Variable(_).to(device)

        optimizer.zero_grad()

        out = lenet(imgs)
        loss = loss_func(out, _)

        loss.backward()
        optimizer.step()

        if i % opt.sample_interval == 0:
            print(
                "[Epoch %d/%d] [Batch %d/%d] [loss: %f]"
                % (epoch, opt.n_epochs, i, len(train_dataloader), loss.item())
            )

    correct = 0.0
    total = 0.0
    for (test_imgs, test_) in test_dataloader:
        test_imgs, test_ = Variable(test_imgs).to(device), Variable(test_).to(device)

        test_out = lenet(test_imgs)
        max_value, predict = torch.max(test_out.data, 1)
        total = total + test_.size(0)
        correct = correct + (predict == test_).sum()

    print('The recognition accuracy of the %d epoch is: %.2f%%' % (epoch, (100.0 * correct / total)))

