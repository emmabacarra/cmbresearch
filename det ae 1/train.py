'''importing packages & necessary functions'''
import os
import sys

# pytorch functionalities
import torch
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F

# data processing
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.data import random_split

# plotting & visualization
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from torchvision.utils import save_image, make_grid

# others
import tqdm
import numpy as np
import webp
from IPython.display import clear_output

from model import ConvVAE
sys.path.append('..')
from functions import net

# gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


'''
======================================================================================================================================
'''


# create a transofrm to apply to each datapoint
transform = transforms.Compose([transforms.ToTensor()])

# download the MNIST datasets
path = '../../Local Data Files/MNIST'
train_dataset = MNIST(path, transform=transform, download=True)
test_dataset  = MNIST(path, transform=transform, download=True)
if __name__ == '__main__':
    # using the same data as testing since we are trying to reproduce the images
    print(type(train_dataset))

# 80-20 train-val split
n_train = int(0.8*len(train_dataset))
n_val = len(train_dataset) - n_train
train_dataset, val_dataset = random_split(train_dataset, [n_train, n_val], generator=np.random.seed(0))
if __name__ == '__main__':
    print(f"Train dataset size: {len(train_dataset)}, Validation dataset size: {len(val_dataset)}, Test dataset size: {len(test_dataset)}")
    print(f"Image size: {train_dataset[0][0].size()}")

# create train, validation and test dataloaders
batch_size = 100
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


img_size = train_dataset[0][0].size()[1]*train_dataset[0][0].size()[2] 


'''
======================================================================================================================================
'''

def loss_function(x, x_hat, mean, log_var, kl_weight=1):
    reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')
    KLD = - 0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp())

    # loss = reconstruction loss + similarity loss (KL divergence)
    return reproduction_loss + kl_weight*KLD

model = ConvVAE(
            image_channels=1,  # setting to 1 since the images are grayscale
            init_channels=8, 
            kernel_size=14,
            padding=12,
            latent_dim=16, 
            leak=0.99, drop=0.01,
            stochastic=False # setting to False makes this deterministic (no sampling) - i.e. a normal autoencoder
        ).to(device)
nnet = net(model, train_loader, val_loader, test_loader, batch_size, linear=False);

optimizer = Adam(model.parameters(), lr=0.001, weight_decay=1e-10);

if __name__ == '__main__':
    nnet.train(optimizer=optimizer, lsfn=loss_function, epochs=35, kl_weight=0.1, live_plot=False, outliers=False)
    torch.save(nnet.model.state_dict(), 'saved_model.pth')
    nnet.evaluate(test_loader)

    latent_dims = (0, 1)
    print(f"Selected latent dimensions: {latent_dims}")

    plt.figure(figsize = (10, 5))

    plt.subplot(1, 2, 1)
    nnet.plat(test_loader, latent_dims) 
    sns.despine()

    plt.subplot(1, 2, 2)
    nnet.prec(test_dataset, n=15, rangex=(-5, 5), rangey=(-5, 5), latent_dims=latent_dims)
    sns.despine()

    nnet.pgen(test_loader, num_images=50)