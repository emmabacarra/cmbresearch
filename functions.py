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
import time

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
        
        self.x_dim = self.trloader.dataset[0][0].size()[1]*self.trloader.dataset[0][0].size()[2]
        self.train_size = len(self.trloader.dataset)

    def train(self, optimizer, lsfn, epochs, kl_weight, live_plot=False, view_interval=100, averaging=True):
        logger = []
        valosses = [] # <-- per epoch
        batch_trlosses = [] # <-- per batch
        batch_ints = self.train_size / (self.batch_size * view_interval)
        absolute_loss = 0

        start_time = time.time()
        for epoch in range(1, epochs+1):
            
            # ========================= training losses =========================
            self.model.train()
            loss_ct, counter = 0, 0
            for i, (batch, _) in enumerate(self.trloader):
                batch_start = time.time()
                counter += 1
                
                if self.linear:
                    batch = batch.view(self.batch_size, self.x_dim)
                batch = batch.to(self.device)

                optimizer.zero_grad()

                outputs, mean, log_var = self.model(batch)
                batch_loss = lsfn(batch, outputs, mean, log_var, kl_weight)
                loss_ct += batch_loss.item()
                absolute_loss += batch_loss.item()

                self.timestamp = time.strftime('%m/%d/%y %H:%M:%S', time.localtime())
                batch_time = time.time() - batch_start
                elapsed_time = time.time() - start_time
                learning_rate = optimizer.param_groups[0]['lr']
                
                batch_log = f'[{self.timestamp}] ({elapsed_time:.2f}s) | Epoch: {epoch} | Batch: {i} ({batch_time:.3f}s) | LR: {learning_rate} | KL Weight: {kl_weight} | Loss: {batch_loss.item()}'
                logger.append((batch_log, batch_time))
                print(batch_log)

                # -------------------------------------------------------------------------------
                if (i+1) % view_interval == 0 or i == len(self.trloader) - 1: # <-- plot for every specified interval of batches (and also account for the last batch)
                    avg_loss = loss_ct / counter
                    if averaging:
                        batch_trlosses.append(avg_loss) # <-- average loss of the interval
                    else:
                        batch_trlosses.append(batch_loss.item())
                    loss_ct, counter = 0, 0 # reset for next interval
                
                    if live_plot: # Plot losses and validation accuracy in real-time 
                        fig, ax = plt.subplots(figsize=(12, 5))
                        clear_output(wait=True)
                        ax.clear()

                        ax.set_title(f'Performance (Epoch {epoch}/{epochs})', weight='bold', fontsize=15)
                        ax.plot(list(range(1, len(batch_trlosses) + 1)), batch_trlosses, 
                                label=f'Training Loss \nLowest: {min(batch_trlosses):.3f} \nAverage: {np.mean(batch_trlosses):.3f} \n', 
                                linewidth=3, color='blue', marker='o', markersize=3)
                        if len(valosses) > 0:
                            ax.plot([i*batch_ints for i in range(1, len(valosses)+1)], valosses, 
                                    label=f'Validation Loss \nLowest: {min(valosses):.3f} \nAverage: {np.mean(valosses):.3f}', 
                                    linewidth = 3, color = 'gold', marker = 'o', markersize = 3)
                        ax.set_ylabel("Loss")
                        ax.set_xlabel(f"Batch Intervals (per {view_interval} batches)")
                        ax.set_xlim(1, len(batch_trlosses) + 1)
                        ax.legend(title = f'Absolute loss: {round(absolute_loss, 3)}', bbox_to_anchor=(1, 1), loc='upper right')

                        plt.show(block=False)
                # -------------------------------------------------------------------------------
                batch_loss.backward()
                optimizer.step()
            
            # ========================= validation losses =========================
            self.model.eval()
            with torch.no_grad():
                tot_valoss = 0
                for batch, _ in self.valoader:

                    if self.linear:
                        batch = batch.view(self.batch_size, self.x_dim)
                    batch = batch.to(self.device)

                    outputs, mean, log_var = self.model(batch)
                    batch_loss = lsfn(batch, outputs, mean, log_var, kl_weight)

                    tot_valoss += batch_loss.item()

                avg_val_loss = tot_valoss / len(self.valoader)
                valosses.append(avg_val_loss)
                
                self.timestamp = time.strftime('%m/%d/%y %H:%M:%S', time.localtime())
                elapsed_time = time.time() - start_time
                learning_rate = optimizer.param_groups[0]['lr']
                
                val_log = f'[{self.timestamp}] ({elapsed_time:.2f}s)  VALIDATION (Epoch {epoch}/{epochs}) | LR: {learning_rate} | KL Weight: {kl_weight} | Loss: {avg_val_loss} -----------'
                logger.append((val_log, None))
                print(val_log)

        end_time = time.time()
        
        self.timestamp = self.timestamp.replace('/', '-').replace(':', '.').replace(' ', '__')
        # -------------------------------------------------------------------------------
        # final plot to account for all tracked losses in an epoch =========================
        fig, ax = plt.subplots(figsize=(12, 5))
        clear_output(wait=True)
        ax.clear()

        ax.set_title(f'Performance (Epoch {epochs}/{epochs})', weight='bold', fontsize=15)
        ax.plot(list(range(1, len(batch_trlosses) + 1)), batch_trlosses, 
                label=f'Training Loss \nLowest: {min(batch_trlosses):.3f} \nAverage: {np.mean(batch_trlosses):.3f} \n', 
                linewidth=3, color='blue', marker='o', markersize=3)
        if len(valosses) > 0:
            ax.plot([i*batch_ints for i in range(1, len(valosses)+1)], valosses, 
                    label=f'Validation Loss \nLowest: {min(valosses):.3f} \nAverage: {np.mean(valosses):.3f}', 
                    linewidth = 3, color = 'gold', marker = 'o', markersize = 3)
        ax.set_ylabel("Loss")
        ax.set_xlabel(f"Batch Intervals (per {view_interval} batches)")
        ax.set_xlim(1, len(batch_trlosses) + 1)
        ax.legend(title = f'Absolute loss: {round(absolute_loss, 3)}', bbox_to_anchor=(1, 1), loc='upper right')

        plt.show(block=False)
        plt.savefig(f"./Loss Plots/{self.timestamp}.png", bbox_inches='tight')
        # -------------------------------------------------------------------------------
        params = [("Model Parameters: ", tuple(self.model.params().items())),
                   ("Encoder Parameters: ", tuple(self.model.encoder.params().items())),
                   ("Decoder Parameters: ", tuple(self.model.decoder.params().items()))]
            
        minutes, seconds = divmod(end_time - start_time, 60)
        print('===========================================================================================',
            '\n===========================================================================================',
            "\n", params[0], "\n", params[1], "\n", params[2], "\n",
            f'\nAbsolute Loss: {absolute_loss:.3f}',
            f'\nTotal Training Time: {int(minutes):02d}m {int(seconds):02d}s | Average Batch Time: {np.mean([log[1] for log in logger if log[1]!=None]):.3f}s')

        with open(f"./Training Logs/{self.timestamp}.txt", "w") as file:
            for param in params:
                file.write(' '.join(map(str, param)) + '\n')
            for log in logger:
                file.write(log[0]+ '\n')
        
        return absolute_loss


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
        self.accuracy = accuracy


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
            plt.title(f"2D Latent Space", fontsize = 15, fontweight = 'bold')
            plt.xlabel(f"Dimension {latent_dims[0]}", fontsize = 12)
            plt.ylabel(f"Dimension {latent_dims[1]}", fontsize = 12)
            if i > self.batch_size:
                plt.colorbar()
                break
            plt.tight_layout()
        plt.show(block=False)
        plt.savefig(f"./Latent Space Plots/{self.timestamp}.png")
    
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
        plt.title(f"Latent Space Images", fontsize = 15, fontweight = 'bold')
        plt.xlabel(f"Dimension {latent_dims[0]}", fontsize = 12)
        plt.ylabel(f"Dimension {latent_dims[1]}", fontsize = 12)
        plt.imshow(img, extent=[*rangex, *rangey])
        plt.show(block=False)
        plt.tight_layout()
        plt.savefig(f"./Latent Space Plots/{self.timestamp}.png")

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
            
            axes[0, 0].set_title(f"Accuracy: {self.accuracy:.3f}", fontsize=15, fontweight='bold', pad=20)
            plt.tight_layout()
            plt.savefig(f"./Generated Samples/{self.timestamp}.png")
            plt.show()