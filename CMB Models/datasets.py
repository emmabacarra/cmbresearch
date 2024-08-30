import os
from torch.utils.data import Dataset
from torchvision.io import read_image

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
        image = read_image(img_path)[:3]  # keep only the first 3 channels, will load image as a tensor
        image = image.float() / 255.0  # Convert torch.uint8 tensor to torch.float32 and normalize to [0, 1] range
        if self.transform:
            image = self.transform(image)
        return image
