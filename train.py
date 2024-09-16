import torch
from torch.utils.data import Dataset, DataLoader
from torchsummary import summary
from torchvision import transforms
import os
from PIL import Image
from models.LeNet_FashionMNIST import LeNet

net = LeNet()

# 使用 summary 函数打印网络结构
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net.to(device)
summary(net, (1, 28, 28))

