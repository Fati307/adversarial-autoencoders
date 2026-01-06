import torch
import torch.nn as nn

BCE = nn.BCELoss()
L1 = nn.L1Loss()
CE = nn.CrossEntropyLoss()
NLL = nn.NLLLoss()

def loss_encoder_generator(reconst, img_data, netD_img, netD_z, z, class_ohe):
    L1_loss = L1(reconst, img_data)
    D_reconst_prob, _ = netD_img(reconst, class_ohe)
    G_img_loss = BCE(D_reconst_prob, torch.ones_like(D_reconst_prob))
    Ez_loss = BCE(netD_z(z, class_ohe), torch.ones_like(D_reconst_prob))
    EG_loss = L1_loss + 0.0001*G_img_loss + 0.01*Ez_loss
    return EG_loss, L1_loss, G_img_loss, Ez_loss

def loss_discriminator_z(netD_z, z_prior, z_fake, class_ohe):
    real_label = torch.ones(z_prior.size(0),1,device=z_prior.device)
    fake_label = torch.zeros(z_fake.size(0),1,device=z_fake.device)
    return BCE(netD_z(z_prior, class_ohe), real_label) + BCE(netD_z(z_fake, class_ohe), fake_label)

def loss_discriminator_img(netD_img, real_img, fake_img, class_map):
    real_label = torch.ones(real_img.size(0),1,device=real_img.device)
    fake_label = torch.zeros(fake_img.size(0),1,device=fake_img.device)
    D_real, _ = netD_img(real_img, class_map)
    D_fake, _ = netD_img(fake_img, class_map)
    return BCE(D_real, real_label) + BCE(D_fake, fake_label)
