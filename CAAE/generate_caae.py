import torch
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from CAAE.utils_caae import one_hot_label_tensor

def generate_letter(netE, netG, test_loader, letter_idx, n=16, n_classes=10, n_z=128, device='cuda'):
    netG.eval()
    netE.eval()
    z_list = []
    with torch.no_grad():
        for imgs, labels in test_loader:
            mask = (labels == letter_idx)
            if mask.sum() > 0:
                imgs_sel = imgs[mask].to(device)
                labels_sel = labels[mask].to(device)
                class_ohe_sel = one_hot_label_tensor(labels_sel, n_classes, device)
                z_sel = netE(imgs_sel, class_ohe_sel)
                z_list.append(z_sel)
            if sum(z.size(0) for z in z_list) >= n:
                break
    z = torch.cat(z_list, dim=0)[:n]
    labels_gen = torch.full((z.size(0),), letter_idx, device=device).long()
    class_ohe_gen = one_hot_label_tensor(labels_gen, n_classes, device)
    with torch.no_grad():
        imgs = netG(z, class_ohe_gen) * 0.5 + 0.5
    grid = vutils.make_grid(imgs, nrow=4, normalize=True)
    plt.figure(figsize=(6,6))
    plt.imshow(grid.permute(1,2,0).cpu())
    plt.axis("off")
    plt.show()
