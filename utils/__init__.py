# utils/__init__.py

from torchvision import transforms

from .dataset import DataLoaderMNIST, DataLoaderNEUSeg
from .train import train_model
from .eval import evaluate_model
from .save import save_model

toTensor = transforms.Compose([transforms.ToTensor()])
toImage = transforms.Compose([transforms.ToPILImage()])
toNumpy = transforms.Compose([transforms.ToTensor()])