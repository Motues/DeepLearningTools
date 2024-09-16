import torch
from torch import nn
from torchsummary import summary

class MLPMNIST(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layer = nn.Sequential(nn.Flatten(), nn.Linear(28 * 28, 64), nn.ReLU())
        self.hidden_layer = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU())
        self.output_layer = nn.Sequential(nn.Linear(64, 10))

    def forward(self, x):
        x = self.input_layer(x)
        x = self.hidden_layer(x)
        x = self.output_layer(x)
        return x

if __name__ == '__main__':
    net = MLPMNIST()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)
    summary(net, (1, 28, 28))