import torch
from torch import nn
from torchsummary import summary


class LeNetMNIST(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(1, 32, kernel_size=3), nn.ReLU())
        self.pool1 = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2))
        self.conv2 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=3), nn.ReLU())
        self.pool2 = nn.Sequential(nn.MaxPool2d(kernel_size=2))
        self.MLP = nn.Sequential(nn.Flatten(), nn.Linear(64 * 5 * 5, 128), nn.Dropout(0.3), nn.ReLU(),
                                 nn.Linear(128, 64), nn.Dropout(0.3), nn.ReLU(), nn.Linear(64, 10))

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.MLP(x)
        return x

class LeNetFashionMNIST(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(1, 32, kernel_size=3), nn.ReLU())
        self.pool1 = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=1))
        self.conv2 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=3), nn.ReLU())
        self.pool2 = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=1))
        self.conv3 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3), nn.ReLU())
        self.pool3 = nn.Sequential(nn.MaxPool2d(kernel_size=3))
        self.MLP = nn.Sequential(nn.Flatten(), nn.Linear(128 * 6 * 6, 200), nn.ReLU(),
                                 nn.Dropout(0.5), nn.Linear(200, 100),nn.ReLU(),
                                 nn.Dropout(0.3), nn.Linear(100, 10))

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.pool3(x)
        x = self.MLP(x)
        return x


if __name__ == '__main__':
    model = LeNetMNIST()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    summary(model, (1, 28, 28))