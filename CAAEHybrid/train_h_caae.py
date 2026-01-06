import torch
from torch import optim
from tqdm import tqdm
from CAAEHybrid.config_h_caae import device, n_classes, n_z, niter, lr_e, lr_g, lr_d, lr_dz
from dataset import train_loader, test_loader
from CAAEHybrid.config_h_caae import data_root, img_size, batchSize
from models.encoder import Encoder
from models.decoder import Generator
from models.discriminator_z import Dz
from models.discriminator_img import Dimg 
from CAAEHybrid.losses_h_caae import BCE, CE, MSE, kl_divergence
from CAAEHybrid.utils_h_caae import weights_init, one_hot_label_tensor, afficher_images


# Initialisation modèles
netE = Encoder().to(device)
netG = Generator().to(device)
netD = Dimg().to(device)
netDz = Dz().to(device)

for net in [netE, netG, netD, netDz]:
    net.apply(weights_init)

optimizerE  = optim.AdamW(netE.parameters(),  lr=lr_e,  betas=(0.5,0.999))
optimizerG  = optim.AdamW(netG.parameters(),  lr=lr_g,  betas=(0.5,0.999))
optimizerD  = optim.AdamW(netD.parameters(),  lr=lr_d,  betas=(0.5,0.999))
optimizerDz = optim.AdamW(netDz.parameters(), lr=lr_dz, betas=(0.5,0.999))


# Entraînement
for epoch in range(1, niter + 1):

    netE.train(); netG.train(); netD.train(); netDz.train()

    losses_epoch = {
        "Dimg": 0, "Dz": 0,
        "Ez": 0, "G": 0,
        "recon": 0, "cls": 0, "kl": 0
    }

    loop = tqdm(train_loader, desc=f"Epoch {epoch}/{niter}")

    for img_data, img_label in loop:

        B = img_data.size(0)
        img_data  = img_data.to(device)
        img_label = img_label.to(device)

        class_ohe = one_hot_label_tensor(img_label, n_classes, device)

        real = torch.ones(B, 1, device=device)
        fake = torch.zeros(B, 1, device=device)

        # 1 Dimg
        optimizerD.zero_grad()

        loss_D_real = BCE(netD(img_data), real)

        z_fake = torch.randn(B, n_z, device=device)
        rand_labels = torch.randint(0, n_classes, (B,), device=device)
        rand_ohe = one_hot_label_tensor(rand_labels, n_classes, device)

        x_fake = netG(z_fake, rand_ohe).detach()
        loss_D_fake = BCE(netD(x_fake), fake)

        loss_Dimg = loss_D_real + loss_D_fake
        loss_Dimg.backward()
        optimizerD.step()

        # 2 Dz
        optimizerDz.zero_grad()

        z_prior = torch.randn(B, n_z, device=device)
        loss_Dz = BCE(netDz(z_prior), real)

        z_enc, _ = netE(img_data)
        loss_Dz += BCE(netDz(z_enc.detach()), fake)

        loss_Dz.backward()
        optimizerDz.step()

        # 3 Encoder + Generator
        optimizerE.zero_grad()
        optimizerG.zero_grad()

        z_enc, class_pred, mu, logvar = netE(img_data, return_var=True)

        loss_Ez = BCE(netDz(z_enc), real)

        z_gen = torch.randn(B, n_z, device=device)
        x_gen = netG(z_gen, rand_ohe)
        loss_G_adv = BCE(netD(x_gen), real)

        x_recon = netG(z_enc, class_ohe)
        loss_recon = MSE(x_recon, img_data)

        loss_cls = CE(class_pred, img_label)
        loss_kl  = kl_divergence(mu, logvar)

        loss_total = (
            loss_Ez +
            loss_G_adv +
            10 * loss_recon +
            loss_cls +
            0.1 * loss_kl
        )

        loss_total.backward()
        optimizerE.step()
        optimizerG.step()

        # Logging
        losses_epoch["Dimg"]  += loss_Dimg.item()
        losses_epoch["Dz"]    += loss_Dz.item()
        losses_epoch["Ez"]    += loss_Ez.item()
        losses_epoch["G"]     += loss_G_adv.item()
        losses_epoch["recon"] += loss_recon.item()
        losses_epoch["cls"]   += loss_cls.item()
        losses_epoch["kl"]    += loss_kl.item()

        loop.set_postfix(
            Dimg=f"{loss_Dimg.item():.3f}",
            Recon=f"{loss_recon.item():.3f}",
            Cls=f"{loss_cls.item():.3f}"
        )

    n = len(train_loader)
    print(
        f"\nEpoch {epoch} | "
        f"Dimg={losses_epoch['Dimg']/n:.4f} | "
        f"Dz={losses_epoch['Dz']/n:.4f} | "
        f"Recon={losses_epoch['recon']/n:.4f} | "
        f"Cls={losses_epoch['cls']/n:.4f} | "
        f"KL={losses_epoch['kl']/n:.4f}"
    )

    # Visualisation des reconstructions vs originales
    if epoch % 5 == 0:
        netE.eval(); netG.eval()
        with torch.no_grad():
            x, y = next(iter(test_loader))
            x, y = x.to(device), y.to(device)
            z, _ = netE(x)
            y_ohe = one_hot_label_tensor(y, n_classes, device)
            x_rec = netG(z, y_ohe)
            afficher_images(x, x_rec, epoch)

print("\n Entraînement terminé")
