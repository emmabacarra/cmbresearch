'''importing packages & necessary functions'''
import sys

import torch

sys.path.append('../')
from datasets import WMAP

# gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

'''
======================================================================================================================================
'''
dataset_path = '../../../Local Data Files/WMAP/Datasets/SkymapK1_9yr_res9_flip_rotate'

stochastic = True  # setting to False makes this deterministic (no sampling) - i.e. a normal autoencoder
batch_size = 100
train_split_percent = 0.8

image_channels=1  # 1 is grayscale, 3 is RGB
init_channels=8
kernel_size=14
stride=2 # must be >0
padding=12

latent_dim=16
leak=0.99
drop=0.01

learning_rate = 1e-6
num_epochs = 10000
save_every_n_epochs = 10
kl_weight = 0.01 # initial
anneal=True
weight_decay = 1e-10

latent_dims = (0, 1)  # dimensions to plot

def get_epochs(): # this is for comparison.ipynb
    return num_epochs

'''
======================================================================================================================================
'''

whole_dataset = WMAP(dataset_path)

# Import and run the generic training script
resume_timestamp = None     # example: '09-05-24__16-24-24'
resume_from_epoch = None    # None if training from scratch/restarting
exec(open('../generic_trainer.py').read())

'''
======================================================================================================================================
09-11-24__13-23-00
- copied from 0-vae__latdim32
- changed leak to 0.7, drop to 0.5
- added stride as a parameter, set to 2

09-11-24__14-24-12
- decreased learning rate to 1e-6
- decreased klw to 0.01

09-11-24__15-30-50
- added kl weight annealing (steady increase to 1) to loss function in generic_trainer.py

09-11-24__20-03-21
- changed latent dimensions from 64 to 8

09-11-24__22-05-07
- added 2 more convolutional layers to the encoder and decoder

09-11-24__22-42-40
- added 4 more convolutional layers to the encoder and decoder
- put nn.Relu() after each of those new convolutional layers

09-11-24__23-01-49
- changed latent dimensions from 8 to 32

09-11-24__23-32-54
- swapped to model 1's parameter settings, trying on model 6's architecture
'''
# debugging

# import os
# sys.path.append('../')
# from functions import plot_histograms

# crop_files = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path)]
# n_histograms = 50
# plot_histograms(crop_files, n_histograms, bins=100)