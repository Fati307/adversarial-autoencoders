import torch
from torch import optim
from tqdm import tqdm
import gc
from dataset import get_dataloaders
from models.encoder import Encoder
from models.decoder import Generator
from models.discriminator_z import Dz
from models.discriminator_img import Dimg
from CAAE.utils_caae import weights_init, afficher_images
from CAAE.losses_caae import loss_encoder_generator, loss_discriminator_z, loss_discriminator_img
from CAAE.config_caae import device, niter, batchSize, img_size, lr_e, lr_g, lr_d, data_root, n_z, n_classes, save_interval, save_path

# 
# 1) Préparer les dataloaders
train_loader, test_loader = get_dataloaders(data_root, img_size, batchSize)


# 2) Initialiser les modèles
netE = Encoder(n_classes).to(device)
netG = Generator().to(device)
netD_img = Dimg().to(device)
netD_z = Dz().to(device)

netE.apply(weights_init)
netG.apply(weights_init)
netD_img.apply(weights_init)
netD_z.apply(weights_init)


# 3) Optimizers
optimizerE = optim.Adam(netE.parameters(), lr=lr_e, betas=(0.5, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr_g, betas=(0.5, 0.999))
optimizerD_img = optim.Adam(netD_img.parameters(), lr=lr_d, betas=(0.5, 0.999))
optimizerD_z = optim.Adam(netD_z.parameters(), lr=lr_d, betas=(0.5, 0.999))

for epoch in range(1, niter+1):
    # Accumulateurs pour l’epoch
    L1_epoch = 0
    Gimg_epoch = 0
    Ez_epoch = 0
    Dz_epoch = 0
    D_epoch = 0

    loop = tqdm(train_loader, total=len(train_loader), desc=f"Epoch {epoch}/{niter}")

    for img_data, img_label in loop:
        img_data = img_data.to(device)
        img_label = img_label.to(device)
        batch_size = img_data.size(0)

        # Création des one-hot labels et cartes
        class_ohe = torch.zeros(batch_size, n_classes, device=device)
        class_ohe.scatter_(1, img_label.view(-1,1), 1)
        class_map = class_ohe.view(batch_size, n_classes, 1, 1).expand(batch_size, n_classes, img_size, img_size)

        # Z aléatoire pour D_z
        z_prior = torch.randn(batch_size, n_z, device=device)

        # 1) Encoder + Generator
        optimizerE.zero_grad()
        optimizerG.zero_grad()
        z = netE(img_data, class_ohe)
        reconst = netG(z, class_ohe)

        EG_loss, L1_loss, G_img_loss, Ez_loss = loss_encoder_generator(
            reconst, img_data, netD_img, netD_z, z, class_map, class_ohe, device
        )

        EG_loss.backward()
        optimizerE.step()
        optimizerG.step()

        # 2) Discriminator Z
        optimizerD_z.zero_grad()
        Dz_loss = loss_discriminator_z(netD_z, z_prior, z.detach(), class_ohe)
        Dz_loss.backward()
        optimizerD_z.step()

        # 3) Discriminator Image
        optimizerD_img.zero_grad()
        D_loss = loss_discriminator_img(netD_img, img_data, reconst.detach(), class_map)
        D_loss.backward()
        optimizerD_img.step()

    
        # Accumulation des losses
        L1_epoch += L1_loss.item()
        Gimg_epoch += G_img_loss.item()
        Ez_epoch += Ez_loss.item()
        Dz_epoch += Dz_loss.item()
        D_epoch += D_loss.item()

        # Mise à jour barre de progression
        loop.set_postfix(
            L1=L1_loss.item(), G_img=G_img_loss.item(),
            Ez=Ez_loss.item(), Dz=Dz_loss.item(), D=D_loss.item()
        )

    # Affichage moyen par epoch

    n_batches = len(train_loader)
    print(f"\nEpoch {epoch}/{niter}")
    print(
        f"L1={L1_epoch/n_batches:.4f} | "
        f"G_img={Gimg_epoch/n_batches:.4f} | "
        f"Ez={Ez_epoch/n_batches:.4f} | "
        f"Dz={Dz_epoch/n_batches:.4f} | "
        f"D={D_epoch/n_batches:.4f}"
    )

    # Reconstruction pour visualisation
    with torch.no_grad():
        sample_data, sample_label = next(iter(test_loader))
        sample_data, sample_label = sample_data.to(device), sample_label.to(device)
        class_ohe_test = torch.zeros(sample_label.size(0), n_classes, device=device)
        class_ohe_test.scatter_(1, sample_label.view(-1,1), 1)
        z_test = netE(sample_data, class_ohe_test)
        reconst_test = netG(z_test, class_ohe_test)
        afficher_images(sample_data, reconst_test, epoch)


    gc.collect()
    torch.cuda.empty_cache()

    # Sauvegarde des modèles
    if epoch % save_interval == 0:
        torch.save(netE.state_dict(), f"{save_path}/netE_epoch_{epoch}.pth")
        torch.save(netG.state_dict(), f"{save_path}/netG_epoch_{epoch}.pth")
        torch.save(netD_img.state_dict(), f"{save_path}/netD_img_epoch_{epoch}.pth")
        torch.save(netD_z.state_dict(), f"{save_path}/netD_z_epoch_{epoch}.pth")

print("Entraînement terminé")
