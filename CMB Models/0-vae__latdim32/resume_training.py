import sys

import torch

sys.path.append('../')
from datasets import WMAP

# gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

'''
======================================================================================================================================
'''
dataset_path = '../../../Local Data Files/WMAP/Datasets/SkymapK1_9yr_res9'

stochastic = True  # setting to False makes this deterministic (no sampling) - i.e. a normal autoencoder
batch_size = 100
train_split_percent = 0.8

image_channels=1  # 1 is grayscale, 3 is RGB
init_channels=8
kernel_size=14
padding=12
latent_dim=32
leak=0.99
drop=0.01

learning_rate = 1e-4
num_epochs = 10000
save_every_n_epochs = 10
kl_weight = 1
weight_decay = 1e-10

latent_dims = (0, 1)  # dimensions to plot

'''
======================================================================================================================================
'''

whole_dataset = WMAP(dataset_path)

checkpoint = torch.load("checkpoint.pth")
model.load_state_dict(checkpoint['model'])
epoch = checkpoint['epoch']

# Import and run the generic training script
exec(open('../generic_trainer.py').read())