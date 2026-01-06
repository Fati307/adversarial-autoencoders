import torch
from torch import nn
from AAE.config_aae import n_channel, n_encode, n_z
class Encoder(nn.Module):
    """Encode image (B, 3, 32, 32) → vecteur latent (B, n_z)"""
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(n_channel, n_encode, 4, 2, 1),      # 32 → 16
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(n_encode, 2*n_encode, 4, 2, 1),    # 16 → 8
            nn.BatchNorm2d(2*n_encode),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(2*n_encode, 4*n_encode, 4, 2, 1),  # 8 → 4
            nn.BatchNorm2d(4*n_encode),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.fc = nn.Linear(4 * n_encode * 4 * 4, n_z)

    def forward(self, x):
        conv = self.conv(x)
        conv_flat = conv.view(x.size(0), -1)
        z = self.fc(conv_flat)
        return z
