import os
from torch.utils.data import Dataset
import numpy as np
# from torchvision.io import read_image, ImageReadMode

class WMAP(Dataset):
    def __init__(self, dataset_path, transform=None, normalize=False):
        self.dataset_path = dataset_path
        self.transform = transform
        self.normalize = normalize
        # List all files in the image directory
        self.img_list = [f for f in os.listdir(dataset_path) if os.path.isfile(os.path.join(dataset_path, f))]

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_path = os.path.join(self.dataset_path, self.img_list[idx])
        image = np.load(img_path)

        if self.normalize == False:
            return image
    
        if self.normalize == True:        
            min_val = np.min(image.flatten())
            max_val = np.max(image.flatten())
            
            if min_val != max_val:
                normalized_image = (image - min_val) / (max_val - min_val)
            else:
                normalized_image = np.zeros_like(image)
            if self.transform:
                normalized_image = self.transform(normalized_image)
        
            return normalized_image