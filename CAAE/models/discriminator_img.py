import torch
from torch import nn
from CAAE.config_caae import n_channel, n_disc
class Dimg(nn.Module):
    """Discriminateur d'image pour 32x32"""
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(n_channel, n_disc, 4, 2, 1),       # 32 -> 16
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(n_disc, n_disc*2, 4, 2, 1),       # 16 -> 8
            nn.BatchNorm2d(n_disc*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(n_disc*2, n_disc*4, 4, 2, 1),     # 8 -> 4
            nn.BatchNorm2d(n_disc*4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(n_disc*4, n_disc*8, 4, 2, 1),     # 4 -> 2
            nn.BatchNorm2d(n_disc*8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.fc = nn.Sequential(
            nn.Linear(n_disc*8*2*2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        h = self.conv(x)
        h = h.view(h.size(0), -1)
        return self.fc(h)