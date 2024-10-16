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
stride=2
padding=12

latent_dim=32
leak=0.99
drop=0.01

learning_rate = 1e-4
num_epochs = 10000
save_every_n_epochs = 10
kl_weight = 1
anneal=False
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
'''
# debugging

# import os
# sys.path.append('../')
# from functions import plot_histograms

# crop_files = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path)]
# n_histograms = 50
# plot_histograms(crop_files, n_histograms, bins=100)