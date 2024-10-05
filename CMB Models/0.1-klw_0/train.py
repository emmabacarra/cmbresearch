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

stochastic = True  # setting to False makes this deterministic (no sampling) - i.e. a normal autoencoder
batch_size = 100
train_split_percent = 0.8

image_channels=1  # 1 is grayscale, 3 is RGB
init_channels=8
kernel_size=14
stride=2
padding=12

latent_dim=32
leak=0.2 # previously 0.99
drop=0.01

learning_rate = 1e-8 # previously 1e-6
num_epochs = 10000
save_every_n_epochs = 1
kl_weight = 0
anneal=True
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
09-12-24__15-34-30
- copied from model 0.0
- decreased kl weight to 0

09-27-24__00-41-08
- changed learning rate from 1e-6 to 1e-8

10-04-24__12-36-05
- removed normalizing images in WMAP dataset class (normalize=False)
- added scale_each=True to make_grid() in debug_plots, normalize=False
- still looks like stark black/white images in debug_plots

10-04-24__13-32-29
- from prior debugging for forward pass preview:
    - set normalize=True in make_grid
    - increased bin size to 200
    - added both log and linear histograms to debug_plots
- swapped to dataset without random flip/rotate applied

temporary:
10-04-24__22-56-10
- latent dim 32 to 128
- removed sigmoid layer in decoder
- anneal = True, changed to epochs/1000
- added plots of latent space, both with PCA and t-SNE
'''