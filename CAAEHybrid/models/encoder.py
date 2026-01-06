import torch
from torch import nn
from CAAEHybrid.config_h_caae import n_channel, n_gen, n_z, n_classes

class Encoder(nn.Module):
    """Encode une image 32x32 → vecteur latent z + prédiction de classe"""
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(n_channel, n_gen, 4, 2, 1),       # 32 -> 16
            nn.BatchNorm2d(n_gen), nn.LeakyReLU(0.2,True),
            nn.Conv2d(n_gen, n_gen*2, 4, 2, 1),        # 16 -> 8
            nn.BatchNorm2d(n_gen*2), nn.LeakyReLU(0.2,True),
            nn.Conv2d(n_gen*2, n_gen*4, 4, 2, 1),      # 8 -> 4
            nn.BatchNorm2d(n_gen*4), nn.LeakyReLU(0.2,True),
            nn.Conv2d(n_gen*4, n_gen*8, 4, 2, 1),      # 4 -> 2
            nn.BatchNorm2d(n_gen*8), nn.LeakyReLU(0.2,True)
        )
        self.fc_mu = nn.Linear(n_gen*8*2*2, n_z)
        self.fc_logvar = nn.Linear(n_gen*8*2*2, n_z)
        self.classifier = nn.Linear(n_z, n_classes)

    def forward(self, x, return_var=False):
        h = self.conv(x).view(x.size(0), -1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        class_pred = self.classifier(z)
        if return_var:
            return z, class_pred, mu, logvar
        return z, class_pred