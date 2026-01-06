import torch
from torch import nn
from AAE.config_aae import n_disc, n_channel
class Dimg (nn.Module):
    """
    Discriminateur d'images pour AAE non-conditionnel
    """
    def __init__(self):
        super(Dimg, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(n_channel, n_disc, 4, 2, 1),   # 32 → 16
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(n_disc, n_disc*2, 4, 2, 1),   # 16 → 8
            nn.BatchNorm2d(n_disc*2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(n_disc*2, n_disc*4, 4, 2, 1), # 8 → 4
            nn.BatchNorm2d(n_disc*4),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.fc = nn.Sequential(
            nn.Linear(n_disc*4*4*4, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        x = self.model(img)
        x = x.view(x.size(0), -1)
        return self.fc(x)