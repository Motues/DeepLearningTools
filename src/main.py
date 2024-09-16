import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torchvision import transforms
import torch.optim as optim
from PIL import Image
from torchvision import utils
import os

class MyData(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.path = os.path.join(root_dir)
        self.image_path = os.listdir(self.path)

    def __getitem__(self, idx):
        image_name = self.image_path[idx]
        label = image_name[0]
        image_item_path = os.path.join(self.root_dir, image_name)
        image_tensor = toTensor(Image.open(image_item_path))
        label_tensor = torch.tensor(int(label))
        return image_tensor, label_tensor

    def __len__(self):
        return len(self.image_path)

class MLP(nn.Module):
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

class LeNet(nn.Module):
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



labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
               'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']



net = torch.load('../model/LeNet_FashionMNIST.pth')
net.eval()

test_dir = "../data/FashionMNIST/test"
toTensor = transforms.Compose([transforms.ToTensor()])
toImage = transforms.Compose([transforms.ToPILImage()])
test_dataset = MyData(test_dir)
test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=True)

correct_num = 0
wrong_num = 0
total_num = 0
step = 0
with torch.no_grad():
    for x, y in test_loader:
        y_hats = net(x)
        for i, y_hat in enumerate(y_hats):
            tar = torch.argmax(y_hat)
            if tar == y[i]:
                correct_num += 1
            else:
                wrong_num += 1
                utils.save_image(x[i], f'../data/FashionMNIST/error/{labels[y[i]]}_{labels[tar]}_{wrong_num}.png')
            total_num += 1
        step += 1
        if step % 100 == 0:
            print(f'accuracy: {float(correct_num / total_num * 100):f} %')
print(f'Final accuracy: {float(correct_num / total_num * 100):f} %')