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

image_channels=1  # setting to 1 since the images are grayscale
init_channels=8
kernel_size=14
stride=2
padding=12

latent_dim=16 # if deterministic set to 16
leak=0.99
drop=0.01

learning_rate = 1e-8
num_epochs = 10000
save_every_n_epochs = 5
kl_weight = 0.1 # =0 if not stochastic else kl_weight
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

10-04-24__15-00-45
- stopped early because of nan values

10-04-24__15-06-51
- decreased learning rate from 1e-4 to 1e-6
- kld was extremely high
- still got nans

10-04-24__15-09-10
- decreased learning rate from 1e-6 to 1e-8
- generated samples still look bad/grainy/unmeaningful
'''