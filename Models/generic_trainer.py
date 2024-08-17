import sys

# pytorch functionalities
import torch
import torch.nn as nn

# data processing
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.data import random_split

# plotting & visualization
import seaborn as sns
import matplotlib.pyplot as plt

from model import ConvVAE
sys.path.append('../')
from functions import experiment

''' ASSUMPTION - the following variables are defined in the main training script:
whole_dataset, batch_size, train_split_percent, device, learning_rate, weight_decay, num_epochs, latent_dims, kl_weight, stochastic, etc.'''

transform = transforms.Compose([transforms.Normalize((0.5,), (0.5,))])
whole_dataset.transform = transform

# train-val split
n_train = int(train_split_percent * len(whole_dataset))
n_val = len(whole_dataset) - n_train
train_subset, val_subset = random_split(whole_dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

if __name__ == '__main__':
    print(f"Train dataset size: {len(train_subset)} | Validation dataset size: {len(val_subset)}")
    print(f"Image size: {train_subset[0][0].size()}")

# create train and validation/test dataloaders
train_loader = DataLoader(dataset=train_subset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(dataset=val_subset, batch_size=batch_size, shuffle=False)

img_size = train_subset[0][0].size()[1] * train_subset[0][0].size()[-1]

def loss_function(x, x_hat, mean, log_var, kl_weight=1):
    x_hat = torch.sigmoid(x_hat) # Sigmoid activation, to change output between 0 and 1 for binary cross entropy
    x = x / 255.0  # Normalize target images if needed
    reconstruction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    return reconstruction_loss + kl_weight * KLD, reconstruction_loss

model = ConvVAE(
    image_channels=image_channels,  # setting to 1 since the images are grayscale
    init_channels=init_channels,
    kernel_size=kernel_size,
    padding=padding,
    latent_dim=latent_dim,
    leak=leak,
    drop=drop,
    stochastic=stochastic
).to(device)

nnet = experiment(model, train_loader, val_loader, batch_size, linear=False)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

def get_model():
    return model

if __name__ == '__main__':
    nnet.train(optimizer=optimizer, lsfn=loss_function, epochs=num_epochs, kl_weight=0 if not stochastic else kl_weight, live_plot=False, outliers=False)
    nnet.evaluate()

    print(f"Selected latent dimensions: {latent_dims}")
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    nnet.plat(latent_dims)
    sns.despine()

    plt.subplot(1, 2, 2)
    nnet.prec(n=15, rangex=(-5, 5), rangey=(-5, 5), latent_dims=latent_dims)
    sns.despine()

    nnet.pgen(num_images=50)
