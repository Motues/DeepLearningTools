import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torchvision import transforms
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

#获取训练和测试数据集
train_dir = "../data/MNIST/train"
test_dir = "../data/MNIST/test"
toTensor = transforms.Compose([transforms.ToTensor()])
train_dataset = MyData(train_dir)
test_dataset = MyData(test_dir)

train_loader = DataLoader(dataset=train_dataset, batch_size=10, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=10, shuffle=True)

#训练网络
net = MLP()
lr = 0.001
Loss = nn.CrossEntropyLoss(reduction='none')
epoch_number = 2
optimizer = optim.Adam(net.parameters(), lr=lr)
#optimizer = optim.SGD(net.parameters(), lr=lr)
step = 0
for epoch in range(epoch_number):
    for x, y in train_loader:
        #x = x.view(-1, 28 *28)
        y_hat = net(x)
        loss = Loss(y_hat, y)
        optimizer.zero_grad()
        loss.sum().backward()
        optimizer.step()
        step += 1
        if step % 500 == 0:
            print(f'loss: {float(loss.sum().item()):f} %')

correct_num = 0
total_num = 0
step = 0
with torch.no_grad():
    for x, y in test_loader:
        #x = x.view(-1, 28 * 28)
        y_hats = net(x)
        for i, y_hat in enumerate(y_hats):
            if torch.argmax(y_hat) == y[i]:
                correct_num += 1
            total_num += 1
        step += 1
        if step % 100 == 0:
            print(f'accuracy: {float(correct_num / total_num * 100):f} %')
print(f'Final accuracy: {float(correct_num / total_num * 100):f} %')

torch.save(net, 'MLP.pth')
