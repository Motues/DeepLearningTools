import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torchvision import transforms, utils
import torch.optim as optim
from PIL import Image
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

class LeNet(nn.Module):
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

train_dir = "../data/MNIST/train"
test_dir = "../data/MNIST/test"
toTensor = transforms.Compose([transforms.ToTensor()])
train_dataset = MyData(train_dir)
test_dataset = MyData(test_dir)

train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=True)

net = LeNet()
lr = 0.001
Loss = nn.CrossEntropyLoss()
epoch_number = 4
optimizer = optim.Adam(net.parameters(), lr=lr)
step = 0
for epoch in range(epoch_number):
    for x, y in train_loader:
        y_hat = net(x)
        loss = Loss(y_hat, y)
        optimizer.zero_grad()
        loss.sum().backward()
        optimizer.step()
        step += 1
        if step % 500 == 0:
            print(f'loss: {float(loss.sum().item()):f} %')
torch.save(net, '../model/LeNet.pth')

net.eval()
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
                utils.save_image(x[i], f'../data/MNIST/error/{y[i]}_{tar}_{wrong_num}.png')
            total_num += 1
        step += 1
        if step % 100 == 0:
            print(f'accuracy: {float(correct_num / total_num * 100):f} %')
print(f'Final accuracy: {float(correct_num / total_num * 100):f} %')


'''
x = torch.rand(size=(1, 1, 28, 28), dtype=torch.float)
for layer in net.children():
    x = layer(x)
    print(layer.__class__.__name__, 'output shape: ', x.shape)
'''