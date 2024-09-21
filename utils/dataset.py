from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch
from PIL import Image
import os
from torch.utils.data import DataLoader


toTensor = transforms.Compose([transforms.ToTensor()])

class DatasetMNIST(Dataset):
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

def data_loader_mnist(root_dir, batch_size, shuffle):
    dataset = DatasetMNIST(root_dir)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

 class DatasetNEUSeg(Dataset):
     def __init__(self, root_dir):
         self.root_dir = root_dir
