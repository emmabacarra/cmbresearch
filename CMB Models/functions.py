import os
from re import search
import numpy as np
from astropy.io import fits
import astropy.visualization as vis
import matplotlib.pyplot as plt
import time
from IPython.display import clear_output


def FitsMapper(files, hdul_index, nrows, ncols, cmap, interpolation, histogram=False, bins=100,
                animation=False, interval=None, stretch=None, 
                vmin=None, vmax=None, contrast=None, bias=None, power=1, percentile=1, time_delay=0.1):
    
    if animation == False:
        total_plots = len(files) * 2  # Each file has an image and a histogram
        ncols = ncols * 2  # Each file needs two columns (one for the image and one for the histogram)
        nrows = (total_plots + ncols - 1) // ncols  # Calculate the required number of rows

        if histogram:
            fig, axs = plt.subplots(nrows, ncols, figsize=(25, 10))  # Adjust the figure size as needed
        else:
            fig, axs = plt.subplots(nrows, ncols, figsize=(15, 10))
        axs = axs.flatten()

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
            ax = axs[i * 2]
            ax.set_title(search('yr\d+', file).group(), weight='bold', fontsize=17)
            ax.imshow(data, cmap=cmap, norm=norm, interpolation=interpolation)
            ax.axis('off')

            hist_ax = axs[i * 2 + 1]
            hist_ax.hist(data.flatten(), bins=bins, color='blue', log=True)
            hist_ax.set_title('Histogram', weight='bold', fontsize=17)
            hist_ax.set_xlabel('Pixel Value')
            hist_ax.set_ylabel('Frequency')
    
    if animation == False:
        plt.tight_layout(pad=0, h_pad=0.2, w_pad=0.2)
        plt.show()

'''
------------------------------------------------------------------------------------------------------------------------------------------
'''
import torch
import torch.nn as nn
import logging
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
import threading
import subprocess
import requests
from scipy.stats import pearsonr
from skimage.metrics import structural_similarity
from skimage.metrics import mean_squared_error
from datetime import timedelta
import random

class experiment:
    def __init__(self, model, trloader, valoader, batch_size):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.trloader = trloader
        self.valoader = valoader
        # self.teloader = teloader
        self.batch_size = batch_size
        
        self.x_dim = self.trloader.dataset[0].shape[0]*self.trloader.dataset[0].shape[1]
        self.train_size = len(self.trloader.dataset) # number of datapoints in the training set
        self.num_batches = self.train_size / self.batch_size # number of batches in the training set

        # self.valosses = []        # <-- per epoch
        self.batch_trlosses = []  # <-- per batch

    def save_checkpoint(self, epoch, optimizer, path='checkpoint.pth'):
        state = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'batch_trlosses': self.batch_trlosses,
            'global_step': self.global_index
            # 'valosses': self.valosses
        }
        torch.save(state, path)

    def load_checkpoint(self, optimizer, path='checkpoint.pth'):
        if os.path.isfile(path):
            checkpoint = torch.load(path)
            self.model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch'] + 1

            # self.valosses.extend(checkpoint['valosses'])
            self.batch_trlosses.extend(checkpoint['batch_trlosses'])
            self.global_index = checkpoint['global_step']
            return start_epoch
        else:
            raise FileNotFoundError(f"No checkpoint found at '{path}'")
    
    
    def open_tensorboard(self, log_dir, port=6006):
        self.port = port
        process = subprocess.Popen(
            f'tensorboard --logdir="{log_dir}" --bind_all --port={port}', 
            shell=True, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE
        )
        while True:
            try:
                response = requests.get(f'http://localhost:{self.port}')
                if response.status_code == 200:
                    break
            except requests.ConnectionError:
                pass

            if process.stdout:
                output = process.stdout.readline().decode('utf-8')
                if output:
                    self.logger.info(output.strip())
            if process.stderr:
                error = process.stderr.readline().decode('utf-8')
                if error:
                    self.logger.error(error.strip())

        self.tensorboard_ready.set()
    
    def loss_function(self, x, x_hat, mean, log_var, kl_weight=1, anneal=False, epoch=None):
        reconstruction_loss = nn.functional.mse_loss(x_hat, x, reduction='mean')
        KL_loss = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
        if anneal:
            kl_weight = min(1.0, epoch / 1000)
        total_loss = reconstruction_loss + kl_weight * KL_loss
        return total_loss, reconstruction_loss, kl_weight, KL_loss

    def train(self, optimizer, epochs, kl_weight, save_every_n_epochs=10,
              averaging=True, resume_from_epoch=None, resume_timestamp=None, anneal=False):  
        
        # ========================== Logger & Tensorboard Configuration ==========================
        torch.backends.cudnn.benchmark = True
        torch.set_printoptions(profile="full")

        if resume_timestamp is not None:
            self.timestamp = resume_timestamp
        else:
            self.timestamp = time.strftime('%m-%d-%y__%H-%M-%S', time.localtime())

        # textual logger
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s %(message)s', datefmt='%m/%d/%y %H:%M:%S')
        os.makedirs(f'./Training Logs/{self.timestamp}', exist_ok=True)

        # file handler
        file_handler = logging.FileHandler(f'./Training Logs/{self.timestamp}/textual.log', mode='a')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        self.logger = logger
        logger.info('Textual logger initiated. Now configuring tensorboard writer.')


        # tensorboard writer
        log_dir = f"./Training Logs/{self.timestamp}/tensorboard"
        writer_train = SummaryWriter(log_dir=log_dir+'/train')
        writer_val = SummaryWriter(log_dir=log_dir+'/validation')
        logger.info(f'Tensorboard writers created. Tensorboard logs will be saved to "{log_dir}".')

        self.tensorboard_ready = threading.Event()
        logger.info('Starting Tensorboard thread...')
        thread = threading.Thread(target=self.open_tensorboard, args=(log_dir,), daemon=True)
        thread.start()

        self.tensorboard_ready.wait()
        logger.info(f'Tensorboard is ready at http://localhost:{self.port}/.')

        logger.info('Continuing to training script...')
        time.sleep(3)
        # ==========================================================================================
        #         ========================== Training Script ==========================
        # ==========================================================================================
        batch_times = []
        self.epoch = 0
        self.epochs = epochs
        start_epoch = 1

        if resume_from_epoch is not None:
            logger.info(f"Resuming training from checkpoint: {resume_from_epoch}.")
            start_epoch = self.load_checkpoint(optimizer, path=f'./Checkpoints/{resume_timestamp}/epoch_{resume_from_epoch}_model.pth')
            logger.info(f"Resumed from epoch {start_epoch}.")

        params = [tuple(self.model.params().items())]
                #   tuple(self.model.encoder.params().items()),
                #   tuple(self.model.decoder.params().items())]
        logger.info(f'Training initiated with the following parameters:'
                    f'\nModel Parameters: {params[0]}')
                    # f'\nEncoder Parameters: {params[1]}'
                    # f'\nDecoder Parameters: {params[2]}\n')
        logger.info(f'Train Loader Image Data Size: {self.trloader.dataset[0].shape}')
        
        start_time = time.time()
        try:
            foldername=f'./Checkpoints/{self.timestamp}'
            os.makedirs(foldername, exist_ok=True)

            for epoch in range(start_epoch, epochs + 1):
                self.epoch = epoch

                # ========================= training losses =========================
                self.model.train()
                randint = random.randint(0, self.num_batches-1) # -1 to avoid index out of range
                for i, batch in enumerate(self.trloader):
                    self.global_index = (epoch-1) * self.num_batches + i
                    batch_start = time.time()

                    batch = batch.view(batch.size(0), 1, 28, 28)
                    batch = batch.to(self.device)

                    optimizer.zero_grad()

                    # debugging -----------------------
                    # if i == randint: 
                    #     writer_train.add_figure(f'Images Before Forward Pass - Batch {i}, Epoch {epoch}',
                    #                              self.debug_plots(batch)[0],
                    #                              global_step=self.global_index)
                    #     writer_train.add_figure(f'Pixel Value Histograms - Batch {i}, Epoch {epoch}',
                    #                             self.debug_plots(batch)[1],
                    #                             global_step=self.global_index)
                    # ---------------------------------

                    outputs, mean, log_var = self.model(batch)
                    batch_loss, reconstruction_loss, klw, kld = self.loss_function(batch, outputs, mean, log_var, kl_weight, anneal, epoch)
                    self.batch_trlosses.append(batch_loss.item())

                    batch_time = time.time() - batch_start
                    elapsed_time = timedelta(seconds=time.time() - start_time)
                    formatted_time = str(elapsed_time).split(".")[0] + f".{int(elapsed_time.microseconds / 10000):02d}"
                    learning_rate = optimizer.param_groups[0]['lr']

                    writer_train.add_scalar('loss', batch_loss.item(), self.global_index)
                    batch_log = f'({formatted_time}) | [{self.epoch}/{epochs}] Batch {i} ({batch_time:.3f}s) | LR: {learning_rate} | KLW: {klw}, KLD (loss): {kld:.3f}, Rec. Loss: {reconstruction_loss:.8f} | Total Loss: {batch_loss.item():.8f}'
                    logger.info(batch_log)
                    batch_times.append(batch_time)
                        
                    batch_loss.backward()
                    optimizer.step()

                
                # ========================= validation losses =========================
                logger.info(f'Calculating validation for epoch {self.epoch}.')
                self.model.eval()
                with torch.no_grad():
                    tot_valoss = 0
                    for batch in self.valoader:

                        batch = batch.view(batch.size(0), 1, 28, 28)
                        batch = batch.to(self.device)
                        
                        outputs, mean, log_var = self.model(batch)
                        batch_loss, reconstruction_loss, klw, kld = self.loss_function(batch, outputs, mean, log_var, kl_weight, anneal, epoch)

                        tot_valoss += batch_loss.item()

                    avg_val_loss = tot_valoss / len(self.valoader)

                    elapsed_time = timedelta(seconds=time.time() - start_time)
                    formatted_time = str(elapsed_time).split(".")[0] + f".{int(elapsed_time.microseconds / 10000):02d}"
                    learning_rate = optimizer.param_groups[0]['lr']

                    writer_val.add_scalar('loss', avg_val_loss, self.global_index)
                    val_log = f'({formatted_time}) | VALIDATION (Epoch {self.epoch}/{epochs}) | LR: {learning_rate} | KLW: {klw}, KLD (loss): {kld:.3f}, Rec. Loss: {reconstruction_loss:.8f} | Total Loss: {avg_val_loss:.8f} -----------'
                    logger.info(val_log)
                
                
                # ========================= Progress Checkpoints =========================
                if self.epoch % save_every_n_epochs == 0 or self.epoch == epochs:
                    self.save_checkpoint(self.epoch, optimizer, path=f'{foldername}/epoch_{self.epoch}_model.pth')
                    logger.info(f'Checkpoint saved for epoch {self.epoch}.')

                    self.evaluate()
                    writer_val.add_figure(f'Generated Samples, Epoch {self.epoch}', self.generate_samples(num_images=50), global_step=self.global_index)
                    logger.info(f'Generated images created for epoch {self.epoch}.')

                    logger.info(f'Creating latent space plot for epoch {self.epoch}...')
                    latent_vectors = self.latent_space()
                    logger.info(f'Latent vectors shape: {latent_vectors.shape}')
                    writer_val.add_embedding(latent_vectors,  tag=f'Latent Space, Epoch {self.epoch}', global_step=self.global_index)
                    logger.info(f'Latent space plot created for epoch {self.epoch}.')

                # clear_output(wait=True)
            end_time = time.time()
            
        except KeyboardInterrupt:
            logger.warning("Training was interrupted by the user.")
            self.save_checkpoint(self.epoch, optimizer, path=f'./Checkpoints/{self.timestamp}/epoch_{self.epoch}_model.pth')
            logger.info(f'Checkpoint saved for epoch {self.epoch}.')
            writer_train.close()
            writer_val.close()

        except (Exception, ValueError, TypeError) as e:
            logger.error(f"An error has occurred: {e}", exc_info=True)
            self.save_checkpoint(self.epoch, optimizer, path=f'./Checkpoints/{self.timestamp}/epoch_{self.epoch}_model.pth')
            logger.info(f'Checkpoint saved for epoch {self.epoch}.')
            
            self.evaluate()
            writer_val.add_figure(f'Generated Samples, Epoch {self.epoch}', self.generate_samples(num_images=50), global_step=self.global_index)
            logger.info(f'Generated images created for epoch {self.epoch}.')
            
            logger.info(f'Creating latent space plot for epoch {self.epoch}...')
            latent_vectors = self.latent_space()
            logger.info(f'Latent vectors shape: {latent_vectors.shape}')
            writer_val.add_embedding(latent_vectors,  tag=f'Latent Space, Epoch {self.epoch}', global_step=self.global_index)
            logger.info(f'Latent space plot created for epoch {self.epoch}.')

            writer_train.close()
            writer_val.close()
            raise

        finally:
            try:
                # end_time = time.time()
                elapsed_time = timedelta(seconds=time.time() - start_time)
                formatted_time = str(elapsed_time).split(".")[0] + f".{int(elapsed_time.microseconds / 10000):02d}"
                
                logger.info(
                    '\n==========================================================================================='
                    '\n===========================================================================================\n'
                   f'----  TRAINING SUMMARY FOR SESSION {self.timestamp}  ----\n'
                   f'\nModel Parameters: {params[0]}'
                #    f'\nEncoder Parameters: {params[1]}'
                #    f'\nDecoder Parameters: {params[2]}\n'
                   f'\nCompleted Epochs: {self.epoch}/{epochs} | Avg Tr.Loss: {np.mean(self.batch_trlosses):.8f}'
                   f'\nTotal Training Time: {formatted_time} | Average Batch Time: {np.mean(batch_times):.3f}s \n'
                )

                writer_train.close()
                writer_val.close()

            except Exception as e:
                logger.error(f"An error has occurred: {e}", exc_info=True)
                writer_train.close()
                writer_val.close()
                raise



    def evaluate(self, threshold=0.1):
        self.model.eval()
        total_correct = 0
        total_pixels = 0

        with torch.no_grad():
            for images in self.valoader:
                images = images.to(self.device)
                images = images.view(images.size(0), 1, 28, 28)

                reconstruction_images, _, _ = self.model(images)
                reconstruction_images = reconstruction_images.view_as(images)

                # Calculate the number of correctly reconstructed pixels
                correct_pixels = (torch.abs(images - reconstruction_images) < threshold).type(torch.float).sum().item()
                total_correct += correct_pixels
                total_pixels += images.numel()
        
        self.accuracy = total_correct / total_pixels
        print(f'Accuracy: {self.accuracy:.3f}')

    
    # plot generated samples
    def generate_samples(self, num_images=10, filesave=False):
        self.model.eval()
        with torch.no_grad():
            data_iter = iter(self.valoader)

            images = next(data_iter)
            images = images[:num_images].to(self.device)
            images = images.view(images.size(0), 1, 28, 28)

            reconstruction_images, _, _ = self.model(images)
            reconstruction_images = reconstruction_images.cpu()

        cols = min(num_images, 5)
        rows = (num_images + cols - 1) // cols
        
        fig = plt.figure(figsize=(20, 4 * rows))
        gridspec = fig.add_gridspec(nrows=rows, ncols=11)

        # Create subplots for original and reconstructed images with distinct background colors
        axes_original = fig.add_subplot(gridspec[:, 0:6], facecolor='lightblue')
        axes_reconstructed = fig.add_subplot(gridspec[:, 6:11])

        # Turn off the axes for the overall subplots
        axes_original.axis('off')
        axes_reconstructed.axis('off')

        for i in range(num_images):
            row = i // cols
            col = i % cols

            img_in = torch.squeeze(images[i]).cpu().numpy()
            img_out = torch.squeeze(reconstruction_images[i]).cpu().numpy()
            # print("img_in, img_out shapes:", img_in.shape, img_out.shape)

            # pearson correlation coefficient and p-value
            pcc, pval = pearsonr(img_in.flatten(), img_out.flatten())

            # structural similarity index
            ssim = structural_similarity(img_in, img_out, win_size=11, data_range=img_out.max() - img_out.min())

            # mean squared error
            mse = mean_squared_error(img_in, img_out)


            # original image before forward pass
            ax1 = fig.add_subplot(gridspec[row, col], facecolor='lightblue')
            ax1.imshow(img_in, cmap='gray', filternorm=False)
            ax1.set_title(f"{i+1}", fontsize=10)
            ax1.set_xlabel(f'Dimensions: {img_in.shape}' 
                           f'\nMin: {img_in.min():.3e}' 
                           f'\nMax: {img_in.max():.3e}', fontsize=7)
            ax1.set_xticks([]), ax1.set_yticks([])

            # reconstructed image after forward pass
            ax2 = fig.add_subplot(gridspec[row, col + 6])
            ax2.imshow(img_out, cmap='gray', filternorm=False)
            ax2.set_title(f"{i+1}", fontsize=10)
            ax2.set_xlabel(f'Dimensions: {img_out.shape}'
                           f'\nMin: {img_out.min():.3e}' 
                           f'\nMax: {img_out.max():.3e}'
                           f'\nPCC: {pcc:.3f} | P-Value: {pval:.3f}' 
                           f'\nMSE: {mse:.3f} | SSIM: {ssim:.3e}', fontsize=7)
            ax2.set_xticks([]), ax2.set_yticks([])

        # Set overall titles for each half
        axes_original.set_title("Original Images", weight='bold', fontsize=15, pad=20)
        axes_reconstructed.set_title("Reconstructed Images", weight='bold', fontsize=15, pad=20)

        # Set the overall title for the entire figure
        fig.suptitle(f"Accuracy: {self.accuracy:.3f}", fontsize=20, fontweight='bold', y=1)

        plt.tight_layout(pad=3)
        if filesave == False:
            return fig
        elif filesave == True:
            os.makedirs('./Generated Samples', exist_ok=True)
            plt.savefig(f"./Generated Samples/{self.timestamp}.png")
        elif isinstance(filesave, str):
            plt.savefig(filesave)
        return fig
    
    def debug_plots(self, batch):
        fig, ax = plt.subplots(figsize=(15, 15))

        # batch shape should be: torch.Size([100, 1, 28, 28])
        img_grid = make_grid(batch, nrow=10, normalize=True, padding=2, scale_each=True) 
        # normalizing for visualization shouldn't affect training
        # needs to also have scale_each=True to normalize each image separately
        # normalize and scale_each need to be True for image to be displayed correctly
        # img_grid shape: torch.Size([3, 302, 302]) <-- dimensions of the grid picture itself
        ax.imshow(img_grid.permute(1, 2, 0).cpu().numpy(), cmap='grey')
        ax.axis("off")

        batch_size, channels, height, width = batch.shape
        ax.set_title(f"Batch Dimensions: {batch.shape}", fontsize=16)

        batch = batch.cpu()
        for i in range(10):
            for j in range(10):
                idx = i * 10 + j
                img = batch[idx]

                # Image dimensions
                height, width = img.shape[1], img.shape[2]
                
                # Min and max pixel values
                min_val, max_val = img.min().item(), img.max().item()

                # Display the image information below the image
                ax.text(
                    j * (width + 2) + width // 2,  # x-position of the text
                    i * (height + 2) + height + 2,    # y-position of the text
                    f"{img.shape}\nmin:{min_val:.2f} max:{max_val:.2f}",
                    fontsize=8, ha='center', va='top', color='white', backgroundcolor='black'
                )
        plt.tight_layout()

        fig2, axs2 = plt.subplots(1, 2, figsize=(12, 6))  # Set the figure size for the histogram
        # dimensions: batch number, channel, height, width
        pixel_values = batch.flatten().numpy()

        axs2[0].hist(pixel_values, bins=200, alpha=0.7, log=True)
        axs2[0].set_title('Pixel Value Histogram (Logarithmic Scale)', fontsize=15, fontweight='bold')
        axs2[0].set_xlabel('value')
        axs2[0].set_ylabel('count')

        axs2[1].hist(pixel_values, bins=200, alpha=0.7, log=False)
        axs2[1].set_title('Pixel Value Histogram (Linear Scale)', fontsize=15, fontweight='bold')
        axs2[1].set_xlabel('value')
        axs2[1].set_ylabel('count')

        return fig, fig2

    # plot latent space
    def latent_space(self):
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE
        fig, axs = plt.subplots(1, 2, figsize=(14, 7))

        latent_vectors = []
        with torch.no_grad():
            for batches in self.valoader:
                batches = batches.view(batches.size(0), 1, 28, 28).to(self.device)
                
                h = self.model.encoder(batches)
                z, mu, logvar = self.model.bottleneck(h)
                latent_vectors.append(mu.cpu().numpy())
            
        latent_vectors = np.concatenate(latent_vectors, axis=0)

        return latent_vectors
    

    # plot reconstructions in latent space
    def prec(self, rangex=(-5, 10), rangey=(-10, 5), n=12, latent_dims = (0, 1)):
        '''
        range in the latent space to generate:
            rangex = range of x values
            rangey = range of y values

        n = number of images to plot
        '''
        w = self.valoader.dataset[0].shape[0]  # image width
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
                x_hat = x_hat.to('cpu').detach().numpy()
                # Convert to single channel if necessary (from when using png with rgb channels)
                if x_hat.shape[0] == 3:
                    x_hat = np.dot(x_hat.transpose(1, 2, 0), [0.2989, 0.5870, 0.1140])
                elif x_hat.shape[0] == 1:
                    x_hat = x_hat.squeeze(0)  # Remove the channel dimension if it's a single channel
                else:
                    continue
                
                img[(n-1-i)*w:(n-1-i+1)*w, j*w:(j+1)*w] = x_hat

        plt.title(f"Latent Space Images", fontsize = 15, fontweight = 'bold')
        plt.xlabel(f"Dimension {latent_dims[0]}", fontsize = 12)
        plt.ylabel(f"Dimension {latent_dims[1]}", fontsize = 12)
        plt.imshow(img, extent=[*rangex, *rangey])
        plt.tight_layout()
        os.makedirs('./Latent Space Plots', exist_ok=True)
        plt.savefig(f"./Latent Space Plots/{self.timestamp}.png")
        
       



'''
------------------------------------------------------------------------------------------------------------------------------------------
'''
import sys
import gc
import importlib.util
class GetModelImages: 
    def __init__(self, path, loader, num_images):
        self.path = path
        self.loader = loader
        self.num_images = num_images
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward_pass(self):
        with torch.no_grad():
            data_iter = iter(self.loader)
            images, _ = next(data_iter)
            images = images[:self.num_images].to(self.device)
            reconstruction_images, _, _ = self.model(images)
            return images.cpu(), reconstruction_images.cpu()
    
    def __enter__(self):
        sys.path.append(self.path)
        spec = importlib.util.spec_from_file_location('train', os.path.join(self.path, 'train.py'))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        self.model = module.get_model().to(self.device)
        self.model.load_state_dict(torch.load(os.path.join(self.path, 'latest_saved_model.pth')))
        # from train import model as model_class
        # self.model = model_class.to(self.device)
        # self.model.load_state_dict(torch.load(f'{self.path}/saved_model.pth'))
        self.model.eval()

        original, reconstructions = self.forward_pass()
        return original, reconstructions, module.get_epochs()

    def __exit__(self, exc_type, exc_value, traceback):
        sys.path.remove(self.path)
        self.model.to('cpu')
        del self.model
        self.model = None
        gc.collect()
        torch.cuda.empty_cache()


'''
------------------------------------------------------------------------------------------------------------------------------------------
'''
def plot_histograms(crop_files, num_histograms=10, bins=50):
    num_histograms = min(num_histograms, len(crop_files))

    cols = min(num_histograms, 3) * 2  
    rows = (num_histograms + cols // 2 - 1) // (cols // 2)

    fig = plt.figure(figsize=(15, 3 * rows))
    gridspec = fig.add_gridspec(nrows=rows, ncols=cols)

    for i in range(num_histograms):
        print(crop_files[i])
        data = np.load(crop_files[i])

        data_flat = data.flatten()

        row = i // (cols // 2)
        col_image = (i % (cols // 2)) * 2
        col_hist = col_image + 1

        ax_image = fig.add_subplot(gridspec[row, col_image])
        ax_image.imshow(data, cmap='gray')
        ax_image.set_title(f"{os.path.basename(crop_files[i])}", fontsize=12, fontweight='bold')
        ax_image.axis('off')

        ax_hist = fig.add_subplot(gridspec[row, col_hist])
        ax_hist.hist(data_flat, bins=bins, color='blue', log=True)
        ax_hist.set_title(f"Histogram", fontsize=12, fontweight='bold')
        ax_hist.set_xlabel('Pixel Value')
        ax_hist.set_ylabel('Frequency')

    plt.tight_layout(pad=2)
    plt.savefig(f'histograms_of_crops_0-{num_histograms}.png')
    print('Finished plotting histograms.')