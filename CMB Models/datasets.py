import os
from torch.utils.data import Dataset
import numpy as np
# from torchvision.io import read_image, ImageReadMode

class WMAP(Dataset):
    def __init__(self, dataset_path, transform=None):
        self.dataset_path = dataset_path
        self.transform = transform
        # List all files in the image directory
        self.img_list = [f for f in os.listdir(dataset_path) if os.path.isfile(os.path.join(dataset_path, f))]

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_path = os.path.join(self.dataset_path, self.img_list[idx])
        # image = read_image(img_path, mode=ImageReadMode.RGB)  
        image = np.load(img_path)
        image = image / 255.0  # normalize to [0, 1] range
        if self.transform:
            image = self.transform(image)
        return image
