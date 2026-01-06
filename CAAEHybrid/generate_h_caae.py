import torch
import matplotlib.pyplot as plt
import torchvision.utils as vutils
from CAAEHybrid.utils_h_caae import one_hot_label_tensor
from CAAEHybrid.config_h_caae import device
from dataset import test_loader
from models.encoder import Encoder
from CAAEHybrid.models.decoder import Generator

def generate_letter(netE, netG, letter_idx, n=16):
    netE.eval(); netG.eval()
    z_list = []
    with torch.no_grad():
        for imgs, labels in test_loader:
            mask = (labels == letter_idx)
            if mask.sum() > 0:
                imgs_sel = imgs[mask].to(device)
                z_sel, _ = netE(imgs_sel)
                z_list.append(z_sel)
            if sum(z.size(0) for z in z_list) >= n: break
    z = torch.cat(z_list, dim=0)[:n]
    class_ohe = one_hot_label_tensor(torch.full((z.size(0),), letter_idx, device=device).long())
    imgs = netG(z, class_ohe) * 0.5 + 0.5
    grid = vutils.make_grid(imgs, nrow=4)
    plt.figure(figsize=(6,6)); plt.imshow(grid.permute(1,2,0).cpu()); plt.axis("off"); plt.show()
