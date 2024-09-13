import torch
import torch.nn as nn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import inspect

class ConvEncoder(nn.Module):
    def __init__(self, image_channels, init_channels, kernel_size, stride, padding, latent_dim, leak, drop):
        super(ConvEncoder, self).__init__()
        
        self.latent_dim = latent_dim
        self.image_channels = image_channels
        self.init_channels = init_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.leak = leak
        self.drop = drop

        self.init_ch_x2 = init_channels * 2        # 16
        self.init_ch_x4 = init_channels * 4        # 32
        self.init_ch_x8 = init_channels * 8        # 64
        self.init_ch_x16= init_channels * 16       # 128

        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, init_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.LeakyReLU(leak),
            nn.Conv2d(init_channels, self.init_ch_x2, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.LeakyReLU(leak),
            nn.Conv2d(self.init_ch_x2, self.init_ch_x4, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.LeakyReLU(leak),
            nn.Conv2d(self.init_ch_x4, self.init_ch_x8, kernel_size=kernel_size, stride=stride, padding=0),
            nn.Flatten(),
            nn.Dropout(drop)
        )

        self.fc = nn.Linear(self.init_ch_x8, self.init_ch_x16)
        self.fc_mean = nn.Linear(self.init_ch_x16, latent_dim) 
        self.fc_var = nn.Linear(self.init_ch_x16, latent_dim)

    def forward(self, x):
        x = self.encoder(x)
        x = self.fc(x)
        mean = self.fc_mean(x)
        variance = self.fc_var(x)
        return mean, variance
    
    def params(self):
            init_params = inspect.signature(self.__init__).parameters
            return {name: getattr(self, name) for name in init_params if name != 'self'}


class ConvDecoder(nn.Module):
    def __init__(self, image_channels, init_channels, kernel_size, stride, padding, latent_dim, leak, drop):
        super(ConvDecoder, self).__init__()

        self.latent_dim = latent_dim
        self.image_channels = image_channels
        self.init_channels = init_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.leak = leak
        self.drop = drop

        self.init_ch_x2 = init_channels * 2        # 16
        self.init_ch_x4 = init_channels * 4        # 32
        self.init_ch_x8 = init_channels * 8        # 64
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(self.init_ch_x8, self.init_ch_x8, kernel_size=kernel_size, stride=stride, padding=0),
            nn.LeakyReLU(leak),
            nn.ConvTranspose2d(self.init_ch_x8, self.init_ch_x4, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.LeakyReLU(leak),
            nn.ConvTranspose2d(self.init_ch_x4, self.init_ch_x2, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.LeakyReLU(leak),
            nn.ConvTranspose2d(self.init_ch_x2, image_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.Sigmoid()
        )

        self.fc = nn.Linear(latent_dim, self.init_ch_x8)

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, self.init_ch_x8, 1, 1) 
        x = self.decoder(x)
        return x
    
    def params(self):
            init_params = inspect.signature(self.__init__).parameters
            return {name: getattr(self, name) for name in init_params if name != 'self'}


class ConvVAE(nn.Module):
    def __init__(self, image_channels=1, init_channels=8, kernel_size=4, stride=2, padding=1, latent_dim=16, leak=0.8, drop=0.03, stochastic=True):
        super(ConvVAE, self).__init__()

        self.latent_dim = latent_dim
        self.stochastic = stochastic

        self.image_channels = image_channels
        self.init_channels = init_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.leak = leak
        self.drop = drop

        self.encoder = ConvEncoder(
                                    image_channels=image_channels, 
                                    init_channels=init_channels, 
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    padding=padding,
                                    latent_dim=latent_dim, 
                                    leak=leak, drop=drop
                                ).to(device)
        self.decoder = ConvDecoder(
                                    image_channels=image_channels, 
                                    init_channels=init_channels, 
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    padding=padding,
                                    latent_dim=latent_dim, 
                                    leak=leak, drop=drop
                                ).to(device)
    
    def reparameterize(self, mean, variance):
        if self.stochastic:
            stdev = torch.randn_like(variance).to(device)
            return mean + variance * stdev
        else:
            return mean

    def forward(self, x):
        x = x.to(device)
        mean, variance = self.encoder(x)
        z_sample = self.reparameterize(mean, variance)
        z = self.decoder(z_sample)
        return z, mean, variance
    
    def params(self):
            init_params = inspect.signature(self.__init__).parameters
            return {name: getattr(self, name) for name in init_params if name != 'self'}