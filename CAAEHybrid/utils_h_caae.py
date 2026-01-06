import torch
import matplotlib.pyplot as plt
import torchvision.utils as vutils
from torch import nn

# Initialisation des poids
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# One-hot encoding labels
def one_hot_label_tensor(labels, n_classes, device):
    return torch.zeros(labels.size(0), n_classes, device=device)\
                .scatter_(1, labels.unsqueeze(1), 1)

# Visualisation images
def afficher_images(real, recon, epoch, nrow=8):
    real = real[:nrow]
    recon = recon[:nrow]

    grid = vutils.make_grid(
        torch.cat([real, recon], dim=0),
        nrow=nrow,
        normalize=True
    )

    plt.figure(figsize=(10, 5))
    plt.title(f"Epoch {epoch} | Haut: Réel | Bas: Reconstruit")
    plt.axis("off")
    plt.imshow(grid.permute(1, 2, 0))
    plt.show()
