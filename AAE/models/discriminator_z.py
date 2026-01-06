import torch
from torch import nn
from AAE.config_aae import n_gen, n_channel, n_z, n_disc
class Dz(nn.Module):
    """Discrimine si z vient de l'encodeur (fake) ou d'une gaussienne (real)"""
    def __init__(self):
        super(Dz, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(n_z, n_disc*8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(n_disc*8, n_disc*4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(n_disc*4, n_disc*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(n_disc*2, 1),
            nn.Sigmoid()
        )

    def forward(self, z):
        return self.model(z)
