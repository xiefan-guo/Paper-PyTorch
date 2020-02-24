import torch


w = torch.empty(3, 5)
torch.nn.init.constant_(w, 8)
x = torch.empty(3, 5)
torch.nn.init.constant_(x, 2)
print(w / x * 5)