import sys

# pytorch functionalities
import torch

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

# train-val split
n_train = int(train_split_percent * len(whole_dataset))
n_val = len(whole_dataset) - n_train
train_subset, val_subset = random_split(whole_dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

if __name__ == '__main__':
    print(f"Train dataset size: {len(train_subset)} | Validation dataset size: {len(val_subset)}")
    print(f"Image size: {train_subset[0].shape}")

# create train and validation/test dataloaders
train_loader = DataLoader(dataset=train_subset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(dataset=val_subset, batch_size=batch_size, shuffle=False)

img_size = train_subset[0].shape[0] * train_subset[0].shape[1]

model = ConvVAE(
    # image_channels=image_channels,  # setting to 1 since the images are grayscale
    # init_channels=init_channels,
    # kernel_size=kernel_size,
    # stride=stride,
    # padding=padding,
    # latent_dim=latent_dim,
    # leak=leak,
    # drop=drop,
    # stochastic=stochastic
).to(device)

nnet = experiment(model, train_loader, val_loader, batch_size)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

def get_model():
    return model

if __name__ == '__main__':
    nnet.train(resume_timestamp=resume_timestamp, resume_from_epoch=resume_from_epoch,
               optimizer=optimizer, anneal=anneal, epochs=num_epochs, 
               kl_weight=0 if not stochastic else kl_weight,
               save_every_n_epochs=save_every_n_epochs)
    # nnet.evaluate()

    # print(f"Selected latent dimensions: {latent_dims}")
    # plt.figure(figsize=(10, 5))

    # plt.subplot(1, 2, 1)
    # nnet.plat(latent_dims)
    # sns.despine()

    # plt.subplot(1, 2, 2)
    # nnet.prec(n=15, rangex=(-5, 5), rangey=(-5, 5), latent_dims=latent_dims)
    # sns.despine()

    # nnet.pgen(num_images=50)
