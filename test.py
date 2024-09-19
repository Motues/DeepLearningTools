from PIL import Image
from utils import toTensor

train_dir = "data/NEU_Seg-main/images/train/000201.jpg"

image = Image.open(train_dir)
image = toTensor(image)

print(image.size())