import torch
from torch import nn
from CAAE.config_caae import n_z
class Dz(nn.Module):
    """Discriminateur sur le latent z"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_z, 256),
            nn.LeakyReLU(0.2, True),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.2, True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    def forward(self, z):
        return self.net(z)