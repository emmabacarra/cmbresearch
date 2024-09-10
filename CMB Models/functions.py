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
import logging

class experiment:
    def __init__(self, model, trloader, valoader, batch_size, linear=True):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.trloader = trloader
        self.valoader = valoader
        # self.teloader = teloader
        self.batch_size = batch_size
        self.linear = linear
        
        self.x_dim = self.trloader.dataset[0].shape[0]*self.trloader.dataset[0].shape[1]
        self.train_size = len(self.trloader.dataset)

        self.valosses = []        # <-- per epoch
        self.batch_trlosses = []  # <-- per batch
        self.absolute_loss = 0

    def save_checkpoint(self, epoch, optimizer, path='checkpoint.pth'):
        state = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'absolute_loss': self.absolute_loss,
            'batch_trlosses': self.batch_trlosses,
            'valosses': self.valosses
        }
        torch.save(state, path)

    def load_checkpoint(self, optimizer, path='checkpoint.pth'):
        if os.path.isfile(path):
            checkpoint = torch.load(path)
            self.model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch'] + 1

            self.valosses.extend(checkpoint['valosses'])
            self.batch_trlosses.extend(checkpoint['batch_trlosses'])
            self.absolute_loss = checkpoint['absolute_loss']
            return start_epoch
        else:
            raise FileNotFoundError(f"No checkpoint found at '{path}'")
    
    def loss_plot(self, epoch, view_interval, saveto=None):
        # Filter out infinite values
        self.batch_trlosses.extend([loss for loss in self.batch_trlosses if np.isfinite(loss)])
        self.valosses.extend([loss for loss in self.valosses if np.isfinite(loss)])

        fig, ax = plt.subplots(figsize=(12, 5))
        clear_output(wait=True)
        ax.clear()

        ax.set_title(f'Performance (Epoch {epoch}/{self.epochs})', weight='bold', fontsize=15)
        ax.plot(list(range(1, len(self.batch_trlosses) + 1)), self.batch_trlosses,
                label=f'Training Loss \nLowest: {min(self.batch_trlosses):.3f} \nAverage: {np.mean(self.batch_trlosses):.3f} \n',
                linewidth=3, color='blue', marker='o', markersize=3)
        if len(self.valosses) > 0:
            ax.plot([i * self.batch_ints for i in range(1, len(self.valosses) + 1)], self.valosses,
                    label=f'Validation Loss \nLowest: {min(self.valosses):.3f} \nAverage: {np.mean(self.valosses):.3f}',
                    linewidth=3, color='gold', marker='o', markersize=3)
        ax.set_ylabel("Loss")
        ax.set_xlabel(f"Batch Intervals (per {view_interval} batches)")
        ax.set_xlim(1, len(self.batch_trlosses) + 1)
        ax.legend(bbox_to_anchor=(1, 1), loc='upper right')

        if saveto != None:
            plt.savefig(saveto, bbox_inches='tight')
    
    def train(self, optimizer, lsfn, epochs, kl_weight, save_every_n_epochs=10, live_plot=False, outliers=True, view_interval=100, averaging=True, resume_from_epoch=None, resume_timestamp=None):  
        
        # ========================== Logger Configuration ==========================
        torch.backends.cudnn.benchmark = True
        torch.set_printoptions(profile="full")

        if resume_timestamp is not None:
            self.timestamp = resume_timestamp
        else:
            self.timestamp = time.strftime('%m-%d-%y__%H-%M-%S', time.localtime())

        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s %(message)s', datefmt='%m/%d/%y %H:%M:%S')

        # file handler
        file_handler = logging.FileHandler(f'./Training Logs/{self.timestamp}.log', mode='a')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # ---------------------------------------------------------------------------
        self.batch_ints = self.train_size / (self.batch_size * view_interval)
        batch_times = []
        self.epoch = 0
        self.epochs = epochs
        start_epoch = 1

        if resume_from_epoch is not None:
            logger.info(f"Resuming training from checkpoint: {resume_from_epoch}.")
            start_epoch = self.load_checkpoint(optimizer, path=f'./Checkpoints/{resume_timestamp}/epoch_{resume_from_epoch}_model.pth')
            logger.info(f"Resumed from epoch {start_epoch}.")

        params = [tuple(self.model.params().items()),
                  tuple(self.model.encoder.params().items()),
                  tuple(self.model.decoder.params().items())]
        logger.info(f'Training initiated with the following parameters:'
                    f'\nModel Parameters: {params[0]}'
                    f'\nEncoder Parameters: {params[1]}'
                    f'\nDecoder Parameters: {params[2]}\n')
        logger.info(f'Train Loader Image Data Size: {self.trloader.dataset[0].shape}')
        
        start_time = time.time()
        try:
            foldername=f'./Checkpoints/{self.timestamp}'
            os.makedirs(foldername, exist_ok=True)

            for epoch in range(start_epoch, epochs + 1):
                self.epoch = epoch

                # ========================= training losses =========================
                self.model.train()
                loss_ct, counter = 0, 0
                for i, batch in enumerate(self.trloader):
                    batch_start = time.time()
                    counter += 1

                    if self.linear:
                        # batch = batch.view(self.batch_size, self.x_dim)
                        batch = batch.view(batch.size(0), -1)
                    else:
                        batch = batch.view(batch.size(0), 1, 28, 28)
                    batch = batch.to(self.device)

                    optimizer.zero_grad()

                    outputs, mean, log_var = self.model(batch)
                    # outputs = torch.sigmoid(outputs)  # <-- Sigmoid activation, to change output between 0 and 1 for binary cross entropy
                    # outputs = torch.clamp(outputs, 0, 1)
                    # batch = batch / 255.0  # <-- Normalize target images if needed
                    batch_loss, reconstruction_loss = lsfn(batch, outputs, mean, log_var, kl_weight)
                    loss_ct += batch_loss.item()
                    self.absolute_loss += batch_loss.item()

                    batch_time = time.time() - batch_start
                    elapsed_time = time.time() - start_time
                    minutes, seconds = divmod(int(elapsed_time), 60)
                    learning_rate = optimizer.param_groups[0]['lr']

                    batch_log = f'({int(minutes)}m {int(seconds):02d}s) | [{self.epoch}/{epochs}] Batch {i} ({batch_time:.3f}s) | LR: {learning_rate} | KLW: {kl_weight}, Rec. Loss: {reconstruction_loss:.3f} | Loss: {batch_loss.item():.4f} | Abs. Loss: {self.absolute_loss:.2f}'
                    logger.info(batch_log)
                    batch_times.append(batch_time)

                    # ---------- Recording Loss ----------
                    if (i + 1) % view_interval == 0 or i == len(self.trloader) - 1:  # <-- plot for every specified interval of batches (and also account for the last batch)
                        avg_loss = loss_ct / counter
                        if (outliers or (not outliers and self.epoch > 1)):
                            if averaging:
                                self.batch_trlosses.append(avg_loss)  # <-- average loss of the interval
                            else:
                                self.batch_trlosses.append(batch_loss.item())
                        else:
                            continue
                        loss_ct, counter = 0, 0  # reset for next interval

                        # FOR REAL-TIME PLOTTING -----------
                        if live_plot:  # Plot losses and validation accuracy in real-time
                            self.loss_plot(self.epoch, view_interval) # absolute_loss
                            
                    batch_loss.backward()
                    optimizer.step()
                
                # ========================= validation losses =========================
                logger.info(f'Calculating validation for epoch {self.epoch}.')
                self.model.eval()
                with torch.no_grad():
                    tot_valoss = 0
                    for batch in self.valoader:

                        if self.linear:
                            # batch = batch.view(self.batch_size, self.x_dim)
                            batch = batch.view(batch.size(0), -1)
                        else:
                            batch = batch.view(batch.size(0), 1, 28, 28)
                        batch = batch.to(self.device)

                        outputs, mean, log_var = self.model(batch)
                        # outputs = torch.sigmoid(outputs)
                        # outputs = torch.clamp(outputs, 0, 1)
                        batch = batch / 255.0
                        batch_loss, reconstruction_loss = lsfn(batch, outputs, mean, log_var, kl_weight)

                        tot_valoss += batch_loss.item()

                    avg_val_loss = tot_valoss / len(self.valoader)
                    self.valosses.append(avg_val_loss)

                    elapsed_time = time.time() - start_time
                    minutes, seconds = divmod(int(elapsed_time), 60)
                    learning_rate = optimizer.param_groups[0]['lr']

                    val_log = f'({int(minutes)}m {int(seconds):02d}s) | VALIDATION (Epoch {self.epoch}/{epochs}) | LR: {learning_rate} | KLW: {kl_weight}, Rec. Loss: {reconstruction_loss:.3f} | Loss: {avg_val_loss:.4f} |  Abs. Loss: {self.absolute_loss:.2f} -----------'
                    logger.info(val_log)
                
                
                # ========================= Progress Checkpoints =========================
                if self.epoch % save_every_n_epochs == 0:
                    self.save_checkpoint(self.epoch, optimizer, path=f'{foldername}/epoch_{self.epoch}_model.pth')
                    logger.info(f'Checkpoint saved for epoch {self.epoch}.')

                    self.loss_plot(self.epoch, view_interval,
                                   saveto=f"{foldername}/epoch_{self.epoch}_loss.png") # absolute_loss
                    logger.info(f'Loss plot saved for epoch {self.epoch}.')

                    self.evaluate()
                    self.pgen(num_images=50, filename=f'{foldername}/epoch_{self.epoch}_samples.png')
                    logger.info(f'Generated images saved for epoch {self.epoch}.')
                

                clear_output(wait=True)
            end_time = time.time()
            
        except KeyboardInterrupt:
            logger.warning("Training was interrupted by the user.")
            self.save_checkpoint(self.epoch, optimizer, path=f'./Checkpoints/{self.timestamp}/epoch_{self.epoch}_model.pth')
            logger.info(f'Checkpoint saved for epoch {self.epoch}.')

        except Exception as e:
            logger.error(f"An error has occurred: {e}", exc_info=True)
            self.save_checkpoint(self.epoch, optimizer, path=f'./Checkpoints/{self.timestamp}/epoch_{self.epoch}_model.pth')
            self.evaluate()
            self.pgen(num_images=50, filename=f'epoch_{self.epoch}_samples.png')
            logger.info(f'Checkpoint saved for epoch {self.epoch}.')
            raise

        finally:
            try:
                end_time = time.time()

                torch.save(self.model.state_dict(), 'latest_saved_model.pth')
                logger.info(f"Model saved as 'saved_model.pth'.")

                minutes, seconds = divmod(end_time - start_time, 60)
                logger.info(
                    '\n==========================================================================================='
                    '\n===========================================================================================\n'
                   f'\nModel Parameters: {params[0]}'
                   f'\nEncoder Parameters: {params[1]}'
                   f'\nDecoder Parameters: {params[2]}\n'
                   f'\nCompleted Epochs: {self.epoch}/{epochs} | Avg Tr.Loss: {np.mean(self.batch_trlosses):.3f} | Absolute Loss: {self.absolute_loss:.3f}'
                   f'\nTotal Training Time: {int(minutes)}m {int(seconds):02d}s | Average Batch Time: {np.mean(batch_times):.3f}s'
                )

                self.loss_plot(self.epoch, view_interval,
                               saveto=f"./Loss Plots/{self.timestamp}.png") # absolute_loss

            except Exception as e:
                logger.error(f"An error has occurred: {e}", exc_info=True)
                raise

        return self.absolute_loss


    def evaluate(self, threshold=0.1):
        self.model.eval()
        total_correct = 0
        total_pixels = 0

        with torch.no_grad():
            for images in self.valoader:
                images = images.to(self.device)
                if self.linear:
                    images = images.view(images.size(0), -1)
                else:
                    images = images.view(images.size(0), 1, 28, 28)
                reconstruction_images, _, _ = self.model(images)
                reconstruction_images = reconstruction_images.view_as(images)

                # Calculate the number of correctly reconstructed pixels
                correct_pixels = (torch.abs(images - reconstruction_images) < threshold).type(torch.float).sum().item()
                total_correct += correct_pixels
                total_pixels += images.numel()
        
        accuracy = total_correct / total_pixels
        print(f'Accuracy: {accuracy:.3f}')
        self.accuracy = accuracy


    # plot latent space
    def plat(self, latent_dims=(0, 1)):
        for i, x in enumerate(self.valoader):
            if self.linear:
                # x = x.view(x.size(0), -1)
                x = x.view(x.size(0), -1)
            else:
                x = x.view(x.size(0), 1, 28, 28)
            x = x.to(self.device)
            z, _ = self.model.encoder(x)
            z = z.to('cpu').detach().numpy()
            if self.model.latent_dim > 2:
                # Select the specified dimensions for plotting
                z_selected = z[:, latent_dims]
            else:
                z_selected = z
        
            plt.scatter(z_selected[:, 0], z_selected[:, 1]) # no c=[labels] or cmap because not sure yet??
            plt.title(f"2D Latent Space", fontsize = 15, fontweight = 'bold')
            plt.xlabel(f"Dimension {latent_dims[0]}", fontsize = 12)
            plt.ylabel(f"Dimension {latent_dims[1]}", fontsize = 12)
            if i > self.batch_size:
                plt.colorbar()
                break
            plt.tight_layout()
        plt.savefig(f"./Latent Space Plots/{self.timestamp}.png")
    
    # plot reconstructions
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
                if self.linear:
                    x_hat = x_hat.reshape(28, 28)
                # else:
                #     x_hat = x_hat.squeeze()  # Remove any singleton dimensions (from when using png with rgb channels)

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
        plt.savefig(f"./Latent Space Plots/{self.timestamp}.png")

    # plot generated samples
    def pgen(self, num_images=10, filename='Default'):
        self.model.eval()
        with torch.no_grad():
            data_iter = iter(self.valoader)
            images = next(data_iter)
            images = images[:num_images].to(self.device)
            if self.linear:
                    images = images.view(images.size(0), -1)
            else:
                images = images.view(images.size(0), 1, 28, 28)
            reconstruction_images, _, _ = self.model(images)
            if self.linear:
                reconstruction_images = reconstruction_images.view(num_images, 1, 28, 28)
            reconstruction_images = reconstruction_images.cpu()

        cols = min(num_images, 5)
        rows = (num_images + cols - 1) // cols
        
        fig = plt.figure(figsize=(15, 3 * rows))
        gridspec = fig.add_gridspec(nrows=rows, ncols=12)

        # Create subplots for original and reconstructed images with distinct background colors
        axes_original = fig.add_subplot(gridspec[:, 0:6], facecolor='lightblue')
        axes_reconstructed = fig.add_subplot(gridspec[:, 6:12])

        # Turn off the axes for the overall subplots
        axes_original.axis('off')
        axes_reconstructed.axis('off')

        for i in range(num_images):
            row = i // cols
            col = i % cols

            ax1 = fig.add_subplot(gridspec[row, col])
            ax2 = fig.add_subplot(gridspec[row, col + 6])

            ax1.imshow(images[i].permute(1, 2, 0).cpu(), cmap='gray')
            ax2.imshow(reconstruction_images[i].permute(1, 2, 0), cmap='gray')
            
            ax1.set_title(f"{i+1}", fontsize=10)
            ax1.axis('off')

            ax2.set_title(f"{i+1}", fontsize=10)
            ax2.axis('off')

        # Set overall titles for each half
        axes_original.set_title("Original Images", weight='bold', fontsize=15, pad=20)
        axes_reconstructed.set_title("Reconstructed Images", weight='bold', fontsize=15, pad=20)

        # Set the overall title for the entire figure
        fig.suptitle(f"Accuracy: {self.accuracy:.3f}", fontsize=20, fontweight='bold', y=1)

        plt.tight_layout()
        if filename == 'Default':
            plt.savefig(f"./Generated Samples/{self.timestamp}.png")
        else:
            plt.savefig(filename)
       



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