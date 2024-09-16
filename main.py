import torch

from model import MLPMNIST
from utils import train_model, evaluate_model, DatasetMNIST
from torch.utils.data import DataLoader


train_dir = "D:/Desktop/Projects/Python Project/MyDeepLearning/data/MNIST/train"
test_dir = "D:/Desktop/Projects/Python Project/MyDeepLearning/data/MNIST/test"
train_dataset = DatasetMNIST(train_dir)
test_dataset = DatasetMNIST(test_dir)

train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=True)

net = MLPMNIST()

train_model(net, train_loader, 0.001, "CrossEntropyLoss", "Adam", 10)
evaluate_model(net, test_loader)
