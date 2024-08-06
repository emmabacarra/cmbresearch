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
sys.path.append('../..')
from functions import experiment

# gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

'''
======================================================================================================================================
'''

stochastic = True  # setting to False makes this deterministic (no sampling) - i.e. a normal autoencoder
batch_size = 100
train_split_percent = 0.8

image_channels=1  # setting to 1 since the images are grayscale
init_channels=8
kernel_size=14
padding=12
latent_dim=16
leak=0.99
drop=0.01

learning_rate = 0.001
num_epochs = 8
kl_weight = 0.1
weight_decay = 1e-10

latent_dims = (0, 1)  # dimensions to plot

'''
======================================================================================================================================
'''

# create a transofrm to apply to each datapoint
transform = transforms.Compose([transforms.ToTensor()])

# download the MNIST datasets
path = '../../Local Data Files/MNIST'
whole_dataset = MNIST(path, transform=transform, download=True)
if __name__ == '__main__':
    # using the same data as testing since we are trying to reproduce the images
    print(type(whole_dataset))

# train-val split
n_train = int(train_split_percent*len(whole_dataset))
n_val = len(whole_dataset) - n_train
train_subset, val_subset = random_split(whole_dataset, [n_train, n_val], generator=np.random.seed(0))
if __name__ == '__main__':
    print(f"Train dataset size: {len(train_subset)} | Validation dataset size: {len(val_subset)}")
    print(f"Image size: {train_subset[0][0].size()}")

# create train and validation/test dataloaders
train_loader = DataLoader(dataset=train_subset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(dataset=val_subset, batch_size=batch_size, shuffle=False)

img_size = train_subset[0][0].size()[1]*train_subset[0][0].size()[2] 

'''
======================================================================================================================================
'''

def loss_function(x, x_hat, mean, log_var, kl_weight=1):
    reconstruction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')
    KLD = - 0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp())

    # loss = reconstruction loss + similarity loss (KL divergence)
    return reconstruction_loss + kl_weight*KLD, reconstruction_loss

model = ConvVAE(
            image_channels=image_channels,  # setting to 1 since the images are grayscale
            init_channels=init_channels, 
            kernel_size=kernel_size,
            padding=padding,
            latent_dim=latent_dim, 
            leak=leak, drop=drop,
            stochastic=stochastic
        ).to(device)
nnet = experiment(model, train_loader, val_loader, batch_size, linear=False);

optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay);

def get_model(): # this is for comparison.ipynb
    return model

if __name__ == '__main__':
    nnet.train(optimizer=optimizer, lsfn=loss_function, epochs=num_epochs, kl_weight=0 if not stochastic else kl_weight, live_plot=False, outliers=False)
    # torch.save(nnet.model.state_dict(), 'saved_model.pth')
    nnet.evaluate()

    print(f"Selected latent dimensions: {latent_dims}")
    plt.figure(figsize = (10, 5))

    plt.subplot(1, 2, 1)
    nnet.plat(latent_dims) 
    sns.despine()

    plt.subplot(1, 2, 2)
    nnet.prec(n=15, rangex=(-5, 5), rangey=(-5, 5), latent_dims=latent_dims)
    sns.despine()

    nnet.pgen(num_images=50)