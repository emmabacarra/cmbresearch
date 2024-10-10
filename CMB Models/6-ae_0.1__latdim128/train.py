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
# dataset_path = '../../../Local Data Files/WMAP/Datasets/SkymapK1_9yr_res9_flip_rotate'
dataset_path = '../../../Local Data Files/WMAP/Datasets/SkymapK1_9yr_res9'

stochastic = False  # setting to False makes this deterministic (no sampling) - i.e. a normal autoencoder
batch_size = 100
train_split_percent = 0.8

# image_channels=1  # 1 is grayscale, 3 is RGB
# init_channels=8
# kernel_size=14
# stride=2
# padding=12

latent_dim=128
# leak=0.2
# drop=0.01

learning_rate = 1e-8
num_epochs = 10000
save_every_n_epochs = 10
kl_weight = 0
anneal=False
weight_decay = 1e-10

latent_dims = (0, 1)  # dimensions to plot

def get_epochs(): # this is for comparison.ipynb
    return num_epochs

'''
======================================================================================================================================
'''

whole_dataset = WMAP(dataset_path, normalize=False)

# Import and run the generic training script
resume_timestamp = None     # example: '09-05-24__16-24-24'
resume_from_epoch = None    # None if training from scratch/restarting
exec(open('../generic_trainer.py').read())

'''
======================================================================================================================================
10-06-24__23-53-05
- copied from model 0.1
- changed latent dimensions to 128
- made deterministic (i.e. autoencoder) --> may not have implemented properly, might need to retrain
- full training, stopped when loss plot plateaued

10-10-24__00-03-19
- adjusted deterministic implementation (should actually be deterministic now)
- exploded to inf loss, so reducing learning rate to 1e-10
'''