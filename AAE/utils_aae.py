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

# One-hot encoding des labels
def one_hot_label_tensor(labels, n_classes, device):
    return torch.zeros(labels.size(0), n_classes, device=device)\
                .scatter_(1, labels.unsqueeze(1), 1)

# Affichage des images (original vs reconstruction)

def afficher_images(original, reconstruit, epoch=None, n_show=8):
    """
    original, reconstruit: tensors (B, C, H, W)
    n_show: nombre d'images à afficher
    """
    original = original[:n_show].cpu()
    reconstruit = reconstruit[:n_show].cpu()

    fig, axes = plt.subplots(2, n_show, figsize=(n_show*2,4))

    for i in range(n_show):
        # Original
        ax = axes[0,i]
        img = original[i]
        if img.shape[0] == 1:
            img = img.squeeze(0)
            ax.imshow(img, cmap='gray')
        else:
            ax.imshow(img.permute(1,2,0))
        ax.axis("off")
        if i==n_show//2 and epoch is not None:
            ax.set_title(f"Epoch {epoch} — Originales")

        # Reconstruction
        ax = axes[1,i]
        img_r = reconstruit[i]
        if img_r.shape[0] == 1:
            img_r = img_r.squeeze(0)
            ax.imshow(img_r, cmap='gray')
        else:
            ax.imshow(img_r.permute(1,2,0))
        ax.axis("off")
        if i==n_show//2 and epoch is not None:
            ax.set_title("Reconstructions")

    plt.tight_layout()
    plt.show()
