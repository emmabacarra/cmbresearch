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

class net:
    def __init__(self, model, trloader, valoader, teloader, batch_size, linear=True):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.trloader = trloader
        self.valoader = valoader
        self.teloader = teloader
        self.batch_size = batch_size
        self.linear = linear

    def train(self, optimizer, epochs, x_dim, lsfn):
        loss_list, val_list = [], []

        for epoch in tqdm.trange(1, epochs+1):
            self.model.train()
            total_train_loss = 0 # <-- per epoch, if inside loader loop it's per batch
            for x, _ in self.trloader:

                if self.linear:
                    x = x.view(self.batch_size, x_dim)
                x = x.to(self.device)

                optimizer.zero_grad()

                outputs, mean, log_var = self.model(x)
                loss = lsfn(x, outputs, mean, log_var)
                
                loss.backward()
                optimizer.step()

                total_train_loss += loss.item()
            
            avg_train_loss = total_train_loss / len(self.trloader)
            loss_list.append(avg_train_loss)
            
            # validation accuracy -------------------------------------------------
            self.model.eval()
            with torch.no_grad():
                total_val_loss = 0
                for x, _ in self.valoader:

                    if self.linear:
                        x = x.view(self.batch_size, x_dim)
                    x = x.to(self.device)

                    outputs, mean, log_var = self.model(x)

                    loss = lsfn(x, outputs, mean, log_var)
                    total_val_loss += loss.item()

                avg_val_loss = total_val_loss / len(self.valoader)
                val_list.append(avg_val_loss)
                
            # Plot losses and validation accuracy in real-time ---------------------
            
            fig, ax = plt.subplots(figsize=(12, 5))
            clear_output(wait=True)

            ax.clear()
            ax.plot(range(1, epoch+1), loss_list, 
                    label=f'Training Loss \nLowest: {min(loss_list):.3f} \nAverage: {np.mean(loss_list):.3f} \n', 
                    linewidth=3, color='blue')
            ax.plot(range(1, epoch+1), val_list, 
                    label=f'Validation Loss \nLowest: {min(val_list):.3f} \nAverage: {np.mean(val_list):.3f}', 
                    linewidth=3, color='gold')
            ax.legend(title=f'Epoch {epoch}/{epochs}', bbox_to_anchor=(1, 1), loc='upper left')
            ax.set_title('Performance')
            ax.set_ylabel("Loss")
            plt.show()


    def evaluate(self, dataloader, threshold=0.1):
        self.model.eval()
        total_correct = 0
        total_pixels = 0

        with torch.no_grad():
            for images, _ in dataloader:
                images = images.to(self.device)
                if self.linear:
                    images = images.view(images.size(0), -1)
                recon_images, _, _ = self.model(images)
                recon_images = recon_images.view_as(images)

                # Calculate the number of correctly reconstructed pixels
                correct_pixels = (torch.abs(images - recon_images) < threshold).type(torch.float).sum().item()
                total_correct += correct_pixels
                total_pixels += images.numel()
        
        accuracy = total_correct / total_pixels
        print(f'Accuracy: {accuracy:.3f}')


    # plot latent space
    def plat(self, data_loader, latent_dims=(0, 1)):
        for i, (x, y) in enumerate(data_loader):
            if self.linear:
                x = x.view(x.size(0), -1)
            x = x.to(self.device)
            z, _ = self.model.encoder(x)
            z = z.to('cpu').detach().numpy()
            if self.model.latent_dim > 2:
                # Select the specified dimensions for plotting
                z_selected = z[:, latent_dims]
            else:
                z_selected = z
        
            plt.scatter(z_selected[:, 0], z_selected[:, 1], c=y, cmap='tab10')
            if i > self.batch_size:
                plt.colorbar()
                break
    
    # plot reconstructions
    def prec(self, data_set, rangex=(-5, 10), rangey=(-10, 5), n=12, latent_dims = (0, 1)):
        '''
        range in the latent space to generate:
            rangex = range of x values
            rangey = range of y values

        n = number of images to plot
        '''
        w = data_set[0][0].size()[1]  # image width
        img = np.zeros((n*w, n*w))
        for i, y in enumerate(np.linspace(*rangey, n)):
            for j, x in enumerate(np.linspace(*rangex, n)):
                if self.model.latent_dim > 2:
                    # Initialize a latent vector with zeros
                    z = torch.zeros((1, self.model.latent_dim)).to(self.device)
                    # Set the chosen dimensions to the corresponding x, y values
                    z[0, latent_dims[0]] = x
                    z[0, latent_dims[1]] = y
                    # Project other dimensions onto this plane with random values
                    remaining_dims = [dim for dim in range(self.model.latent_dim) if dim not in latent_dims]
                    z[0, remaining_dims] = torch.randn(len(remaining_dims)).to(self.device)
                else:
                    z = torch.Tensor([[x, y]]).to(self.device)
                
                x_hat = self.model.decoder(z)
                if self.linear:
                    x_hat = x_hat.reshape(28, 28)
                x_hat = x_hat.to('cpu').detach().numpy()
                img[(n-1-i)*w:(n-1-i+1)*w, j*w:(j+1)*w] = x_hat
    
        plt.imshow(img, extent=[*rangex, *rangey])

    def pgen(self, dataloader, num_images=10):
        self.model.eval()
        with torch.no_grad():
            data_iter = iter(dataloader)
            images, _ = next(data_iter)
            images = images[:num_images].to(self.device)
            if self.linear:
                 images = images.view(num_images, -1)
            recon_images, _, _ = self.model(images)
            if self.linear:
                recon_images = recon_images.view(num_images, 1, 28, 28)
            recon_images = recon_images.cpu()

            fig, axes = plt.subplots(2, num_images, figsize=(15, 3), sharex=True, sharey=True)

            for i in range(num_images):
                ax1 = axes[0, i]
                ax2 = axes[1, i]

                ax1.imshow(images[i].view(28, 28).cpu(), cmap='gray')
                if i == 0:
                    ax1.set_ylabel("Original", weight='bold', fontsize=11)
                    ax1.set_xticks([])
                    ax1.set_yticks([])
                else:
                    ax1.axis('off')

                ax2.imshow(recon_images[i].view(28, 28), cmap='gray')
                if i == 0:
                    ax2.set_ylabel("Reconstructed", weight='bold', fontsize=11)
                    ax2.set_xticks([])
                    ax2.set_yticks([])
                else:
                    ax2.axis('off')
            
            plt.tight_layout()
            plt.show()