import torch
import matplotlib.pyplot as plt
import torchvision.utils as vutils
from torch import nn

# Initialisation des poids
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0)
    elif classname.find('BatchNorm') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0)

# One-hot encoding labels
def one_hot_label_tensor(labels, n_classes, device):
    return torch.zeros(labels.size(0), n_classes, device=device)\
                .scatter_(1, labels.unsqueeze(1), 1)

# Visualisation images
def afficher_images(original, reconstruit, epoch, n_show=8):
    original = original[:n_show].cpu()
    reconstruit = reconstruit[:n_show].cpu()
    fig, axes = plt.subplots(2, n_show, figsize=(n_show*2, 4))
    for i in range(n_show):
        ax = axes[0, i]
        ax.imshow(original[i].squeeze()*0.5+0.5, cmap='gray')
        ax.axis("off")
        if i == n_show//2 and epoch is not None:
            ax.set_title(f"Epoch {epoch} — Originales")
        ax = axes[1, i]
        ax.imshow(reconstruit[i].squeeze()*0.5+0.5, cmap='gray')
        ax.axis("off")
        if i == n_show//2 and epoch is not None:
            ax.set_title("Reconstructions")
    plt.tight_layout()
    plt.show()
