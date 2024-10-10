import torch
import torch.nn as nn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import inspect

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class UnFlatten(nn.Module):
    def forward(self, input, size=1024):
        return input.view(input.size(0), size, 1, 1)

class ConvVAE(nn.Module):
    def __init__(self, image_channels=1, h_dim=1024, latent_dim=128, stochastic=False):
        super(ConvVAE, self).__init__()

        self.image_channels = image_channels
        self.h_dim = h_dim
        self.latent_dim = latent_dim
        self.stochastic = stochastic

        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, 32, kernel_size=3, stride=2, padding=1),  # 28x28 -> 14x14
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 14x14 -> 7x7
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 7x7 -> 4x4
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1),  # 4x4 -> 2x2
            nn.ReLU(),
            Flatten()
        )
        
        self.fc1 = nn.Linear(h_dim, latent_dim)
        self.fc2 = nn.Linear(h_dim, latent_dim)
        self.fc3 = nn.Linear(latent_dim, h_dim)
        
        self.decoder = nn.Sequential(
            UnFlatten(),
            nn.ConvTranspose2d(h_dim, 128, kernel_size=3, stride=2),  # 1x1 -> 3x3
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2),  # 3x3 -> 7x7
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # 7x7 -> 14x14
            nn.ReLU(),
            nn.ConvTranspose2d(32, image_channels, kernel_size=4, stride=2, padding=1)  # 14x14 -> 28x28
            # nn.Sigmoid(),
        )

    def bottleneck(self, h):
        mu, logvar = self.fc1(h), self.fc2(h)

        # reparameterization:
        std = logvar.mul(0.5).exp_()
        esp = torch.randn(*mu.size()).to(device)
        z = mu + std * esp

        return z, mu, logvar

    def forward(self, x):
        x = x.to(device)

        # encoder
        h = self.encoder(x).to(device)
        z, mu, logvar = self.bottleneck(h)

        # decoder
        if self.stochastic: # VAEs (stochastic)
            z = self.fc3(z)
            z = self.decoder(z).to(device)
        else: # AEs (deterministic)
            z = self.fc3(mu)
            z = self.decoder(mu).to(device)

        return z, mu, logvar
    
    def params(self):
            init_params = inspect.signature(self.__init__).parameters
            return {name: getattr(self, name) for name in init_params if name != 'self'}