from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch
from PIL import Image
import os
from torch.utils.data import DataLoader
import numpy as np


toTensor = transforms.Compose([transforms.ToTensor()])

class DatasetMNIST(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.path = os.path.join(root_dir)
        self.image_list = os.listdir(self.path)

    def __getitem__(self, idx):
        image_name = self.image_list[idx]
        label = image_name[0]
        image_item_path = os.path.join(self.root_dir, image_name)
        image_tensor = toTensor(Image.open(image_item_path))
        label_tensor = torch.tensor(int(label))
        return image_tensor, label_tensor

    def __len__(self):
        return len(self.image_path)

def DataLoaderMNIST(root_dir, batch_size, shuffle):
    dataset = DatasetMNIST(root_dir)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

class DatasetNEUSeg(Dataset):
     def __init__(self, root_dir, type_dir):
         self.root_dir = root_dir
         self.type_dir = type_dir
         self.images_path = os.path.join(root_dir, 'images', type_dir)
         self.images_list = os.listdir(self.images_path)
         self.annotations_path = os.path.join(root_dir, 'annotations', type_dir)
         self.annotations_list = os.listdir(self.annotations_path)

     def __getitem__(self, idx):
         image_name = self.images_list[idx]
         label_name = self.annotations_list[idx]
         image_item_path = os.path.join(self.root_dir, 'images', self.type_dir, image_name)
         label_item_path = os.path.join(self.root_dir, 'annotations', self.type_dir, label_name)
         image_tensor = toTensor(Image.open(image_item_path))
         label_tensor = toTensor(Image.open(label_item_path))
         return image_tensor, label_tensor

     def __len__(self):
         return len(self.images_list)

def DataLoaderNEUSeg(root_dir, batch_size, shuffle, type_dir):
    dataset = DatasetNEUSeg(root_dir, type_dir)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)