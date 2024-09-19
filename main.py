import torch

from model import MLPMNIST
from utils import train_model, evaluate_model, DataLoaderMNIST


train_dir = "D:/Desktop/Projects/Python Project/MyDeepLearning/data/MNIST/train"
test_dir = "D:/Desktop/Projects/Python Project/MyDeepLearning/data/MNIST/test"

train_loader = DataLoaderMNIST(train_dir, 32, True)
test_loader = DataLoaderMNIST(test_dir, 32, False)


net = MLPMNIST()

train_model(net, train_loader, 0.001, "CrossEntropyLoss", "Adam", 10)
evaluate_model(net, test_loader)
