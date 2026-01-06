import torch
from torch import nn
from AAE.config_aae import n_gen, n_channel, n_z
class Generator(nn.Module):
    """Decode z → image (B, 3, 32, 32)"""
    def __init__(self):
        super(Generator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(n_z, 8 * n_gen * 4 * 4),
            nn.BatchNorm1d(8 * n_gen * 4 * 4),
            nn.ReLU(True)
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(8*n_gen, 4*n_gen, 4, 2, 1),   # 4 → 8
            nn.BatchNorm2d(4*n_gen),
            nn.ReLU(True),
            nn.ConvTranspose2d(4*n_gen, 2*n_gen, 4, 2, 1),   # 8 → 16
            nn.BatchNorm2d(2*n_gen),
            nn.ReLU(True),
            nn.ConvTranspose2d(2*n_gen, n_channel, 4, 2, 1), # 16 → 32
            nn.Tanh()
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, 8*n_gen, 4, 4)
        img = self.deconv(x)
        return img
