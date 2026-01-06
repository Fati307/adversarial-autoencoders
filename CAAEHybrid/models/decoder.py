import torch
from torch import nn
from CAAEHybrid.config_h_caae import  n_channel, n_gen, n_z, n_classes

class Generator(nn.Module):
    """Générateur: z + one-hot → image 32x32"""
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(n_z+n_classes, n_gen*8*2*2),
            nn.BatchNorm1d(n_gen*8*2*2),
            nn.ReLU(True)
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(n_gen*8, n_gen*4, 4, 2, 1),  # 2 -> 4
            nn.BatchNorm2d(n_gen*4), nn.ReLU(True),
            nn.ConvTranspose2d(n_gen*4, n_gen*2, 4, 2, 1),  # 4 -> 8
            nn.BatchNorm2d(n_gen*2), nn.ReLU(True),
            nn.ConvTranspose2d(n_gen*2, n_gen, 4, 2, 1),    # 8 -> 16
            nn.BatchNorm2d(n_gen), nn.ReLU(True),
            nn.ConvTranspose2d(n_gen, n_channel, 4, 2, 1),  # 16 -> 32
            nn.Tanh()
        )

    def forward(self, z, y):
        h = torch.cat([z, y], dim=1)
        h = self.fc(h).view(-1, n_gen*8, 2, 2)
        return self.deconv(h)