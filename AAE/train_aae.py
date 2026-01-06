import torch
from AAE.config_aae import device, niter, batchSize, img_size, lr_e, lr_g, lr_d, data_root, n_z
from torch import optim, nn
from torch.utils.data import DataLoader, random_split
import torchvision.datasets as dset
import torchvision.transforms as transforms
from models.encoder import Encoder
from models.decoder import Generator
from models.discriminator_z import Dz
from models.discriminator_img import Dimg
from AAE.utils_aae import afficher_images, weights_init  
from AAE.losses_aae import loss_encoder_generator, loss_discriminator_z, loss_discriminator_img
from tqdm import tqdm
import matplotlib.pyplot as plt
import gc
from dataset import get_dataloaders
train_loader, test_loader = get_dataloaders(data_root, img_size=img_size, batch_size=batchSize)

# Initialisation des modèles
netE = Encoder().to(device)
netG = Generator().to(device)
netD_img = Dimg().to(device)
netD_z = Dz().to(device)

# Appliquer l'initialisation des poids
netE.apply(weights_init)
netG.apply(weights_init)
netD_img.apply(weights_init)
netD_z.apply(weights_init)

# Optimizers
optimizerE = optim.Adam(netE.parameters(), lr=lr_e, betas=(0.5,0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr_g, betas=(0.5,0.999))
optimizerD_img = optim.Adam(netD_img.parameters(), lr=lr_d, betas=(0.5,0.999))
optimizerD_z = optim.Adam(netD_z.parameters(), lr=lr_d, betas=(0.5,0.999))

BCE = nn.BCELoss()
L1 = nn.L1Loss()

print(f"Configuration: {niter} epochs, batch_size={batchSize}, img_size={img_size}")
print("INNOVATION: AAE non-conditionnel (pas de labels)")

for epoch in range(1, niter+1):
    L1_epoch = 0
    Gimg_epoch = 0
    Ez_epoch = 0
    Dz_epoch = 0
    D_epoch = 0

    loop = tqdm(train_loader, total=len(train_loader), desc=f"Epoch {epoch}/{niter}")

    for img_data, _ in loop:
        img_data = img_data.to(device)
        batch_size = img_data.size(0)

        # ----------------------------
        # Label smoothing
        # ----------------------------
        real_label = torch.ones(batch_size, 1, device=device) * 0.9
        fake_label = torch.zeros(batch_size, 1, device=device) + 0.1
        z_prior = torch.randn(batch_size, n_z, device=device)

         # 1) Encoder + Generator
        optimizerE.zero_grad()
        optimizerG.zero_grad()

        z = netE(img_data)
        reconst = netG(z)

        EG_loss, L1_loss, G_img_loss, Ez_loss = loss_encoder_generator(reconst, img_data, netD_img, netD_z, z, real_label)
        EG_loss.backward()
        optimizerE.step()
        optimizerG.step()

        # 2) Discriminateur z
        optimizerD_z.zero_grad()
        Dz_loss = loss_discriminator_z(netD_z, z_prior, z.detach(), real_label, fake_label)
        Dz_loss.backward()
        optimizerD_z.step()

        # 3) Discriminateur images
        optimizerD_img.zero_grad()
        D_loss = loss_discriminator_img(netD_img, img_data, reconst.detach(), real_label, fake_label)
        D_loss.backward()
        optimizerD_img.step()

        # Accumulateurs
        L1_epoch += L1_loss.item()
        Gimg_epoch += G_img_loss.item()
        Ez_epoch += Ez_loss.item()
        Dz_epoch += Dz_loss.item()
        D_epoch += D_loss.item()

        loop.set_postfix(L1=L1_loss.item(), G_img=G_img_loss.item(),
                         Ez=Ez_loss.item(), Dz=Dz_loss.item(), D=D_loss.item())

    # Moyennes par epoch
    n_batches = len(train_loader)
    print(f"\nEpoch {epoch}/{niter}")
    print(f"L1={L1_epoch/n_batches:.4f} | "
          f"G_img={Gimg_epoch/n_batches:.4f} | "
          f"Ez={Ez_epoch/n_batches:.4f} | "
          f"Dz={Dz_epoch/n_batches:.4f} | "
          f"D={D_epoch/n_batches:.4f}")

    # ----------------------------
    # Affichage reconstruction
    # ----------------------------
    with torch.no_grad():
        sample_data, _ = next(iter(test_loader))
        sample_data = sample_data.to(device)
        z_test = netE(sample_data)
        reconst_test = netG(z_test)
        afficher_images(sample_data, reconst_test, epoch)

    # nettoyage mémoire
    gc.collect()
    torch.cuda.empty_cache()