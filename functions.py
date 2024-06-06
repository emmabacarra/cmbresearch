import os
from re import search
import numpy as np
from astropy.io import fits
import astropy.visualization as vis
import matplotlib.pyplot as plt
import time
from IPython.display import display, HTML, Image, clear_output


def FitsMapper(files, hdul_index, nrows, ncols, cmap, interpolation, 
                animation=False, interval=None, stretch=None, 
                vmin=None, vmax=None, contrast=None, bias=None, power=1, percentile=1, time_delay=0.1):
    
    if animation == False:
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 9))

    for i, file in enumerate(files):
        with fits.open(file) as hdul:
            data = hdul[hdul_index].data
        
        # creating dictionaries for available intervals and stretches in astropy.visualization (vis in this notebook)
        intervals = {'ZScale': vis.ZScaleInterval(),
                    'MinMax': vis.MinMaxInterval(),
                    'Percentile': vis.PercentileInterval(percentile),
                    'AsymPercentile': vis.AsymmetricPercentileInterval(vmin, vmax),
                    'Manual': vis.ManualInterval(vmin=vmin, vmax=vmax)}
        
        stretches = {'Linear': vis.LinearStretch(),
                    'Asinh': vis.AsinhStretch(),
                    'Log': vis.LogStretch(),
                    'Sqrt': vis.SqrtStretch(),
                    'Hist' : vis.HistEqStretch(data),
                    'Power': vis.PowerStretch(power),
                    'Sinh': vis.SinhStretch(),
                    'Contrast': vis.ContrastBiasStretch(contrast=contrast, bias=bias)}

        # converting data to float type and normalizing
        data = np.nan_to_num(data.astype(float))
        vis_vmin, vis_vmax = np.percentile(data, [vmin, vmax])
        norm = vis.ImageNormalize(data, vmin=vis_vmin, vmax=vis_vmax, interval=intervals[interval], stretch=stretches[stretch])

        if animation == True:
            fig, ax = plt.subplots()
            ax.set_title(search('yr\d+', file).group(), weight='bold', fontsize=17)
            ax.imshow(data, cmap=cmap, norm=norm, interpolation=interpolation)
            ax.axis('off')
            plt.tight_layout(pad=0, h_pad=0, w_pad=2)
            plt.show()

            time.sleep(time_delay) # need to fix function architecture to remove lag from loading/manipulating data
            if file != files[-1]:
                clear_output(wait=True)
        else:
            ax = axs[i // 3, i % 3]
            ax.set_title(search('yr\d+', file).group(), weight='bold', fontsize=17)
            ax.imshow(data, cmap=cmap, norm=norm, interpolation=interpolation)
            ax.axis('off')
    
    if animation == False:
        plt.tight_layout(pad=0, h_pad=0.2, w_pad=0.2)
        plt.show()

'''
------------------------------------------------------------------------------------------------------------------------------------------
Under Construction: CMBnet class for handling data and training neural networks
'''

# pytorch functionalities
import torch
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F

# data processing
from astropy.io import fits
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torch.utils.data import random_split
from sklearn.utils import shuffle

# plotting & visualization
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import ImageGrid
from torchvision.utils import save_image, make_grid

# others
import tqdm
import numpy as np
import webp
from IPython.display import clear_output

class CMBnet:
    def __init__(self, model, trloader, valoader):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.trloader = trloader
        self.valoader = valoader

    
    def train(loss_fn, optimizer, epochs, x_dim):

        for epoch in tqdm.trange(epochs):
            self.model.train()
            for x in self.trloader:
                loss_ct = 0
                x = x.view(-1, x_dim).to(self.device)

                optimizer.zero_grad()

                outputs, mean, log_var = self.model(x)
                loss = loss_fn(x, outputs, mean, log_var)
                
                loss.backward()
                loss_ct += loss.item()
                avg_loss = loss_ct / len(self.trloader)
                loss_list.append(avg_loss)
                optimizer.step()
            
            # validation accuracy -------------------------------------------------
            self.model.eval()
            with torch.no_grad():

                loss_ct = 0
                for x in self.valoader:
                    x = x.view(-1, x_dim).to(device)
                    outputs, mean, log_var = self.model(x)

                    loss = loss_fn(x, outputs, mean, log_var)
                    loss_ct += loss.item()

                    # correct = (torch.argmax(outputs, dim=1).to(device) == torch.argmax(x, dim=1)).type(torch.FloatTensor)
                    # val_list.append(correct.mean())

                avg_loss = loss_ct / len(train_loader)
                val_list.append(avg_loss)
                

            # Plot losses and validation accuracy in real-time ---------------------
            if epoch > 0:
                fig, ax = plt.subplots(figsize=(12, 5))
                clear_output(wait=True)

                ax.clear()
                ax.plot(loss_list, label='Training Loss', linewidth = 3, color = 'blue')
                ax.plot([i*len(train_loader) for i in range(epoch+1)], val_list, label='Validation Loss', linewidth = 3, color = 'gold')
                ax.legend(title=f'Lowest Loss: {min(loss_list):.3f} \nAverage Loss: {np.mean(loss_list):.3f}', bbox_to_anchor=(1.25, 1))
                ax.set_title('Training Performance')
                ax.set_ylabel("Loss")
                plt.show()