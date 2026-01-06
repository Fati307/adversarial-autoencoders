import torch
import matplotlib.pyplot as plt
import torchvision.utils as vutils
from models.decoder import Generator
from AAE.config_aae import device, n_z

def generate_images(netG, n=16):
    netG.eval()
    with torch.no_grad():
        z = torch.randn(n, n_z, device=device)
        imgs = netG(z)
        imgs = imgs * 0.5 + 0.5
        grid = vutils.make_grid(imgs, nrow=4, normalize=True)
        plt.figure(figsize=(6,6))
        plt.imshow(grid.permute(1,2,0).cpu())
        plt.axis("off")
        plt.title("Images générées")
        plt.show()
# Usage example:
# generate_images(netG, n=16)