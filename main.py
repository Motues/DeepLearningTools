import torch
from torchsummary import summary

from model import YOLO11_seg
from utils import DataLoaderNEUSeg
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = YOLO11_seg(nc=80)
model = model.to(device)

x = torch.randn(1, 3, 256, 256).to(device)
y = model(x)
for item in y:
    print(type(item))
    if isinstance(item, torch.Tensor):
        print(item.shape)
    elif isinstance(item, list):
        print(len(item))
        for i in item:
            print(i.shape)

