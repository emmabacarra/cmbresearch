import os
from torch.utils.data import Dataset
from torchvision.io import read_image

class WMAP(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        # List all files in the image directory
        self.img_list = [f for f in os.listdir(img_dir) if os.path.isfile(os.path.join(img_dir, f))]

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_list[idx])
        image = read_image(img_path)  # Loads the image as a tensor
        image = image.float() / 255.0  # Convert torch.uint8 tensor to torch.float32 and normalize to [0, 1] range
        if self.transform:
            image = self.transform(image)
        return image
