import torch
import torch.nn as nn

BCE = nn.BCELoss()
L1 = nn.L1Loss()

def loss_encoder_generator(reconst, img_real, netD_img, netD_z, z, real_label):
    """
    Calcul de la perte pour Encoder + Generator
    """
    L1_loss = L1(reconst, img_real)
    G_img_loss = BCE(netD_img(reconst), real_label)
    Ez_loss = BCE(netD_z(z), real_label)
    EG_loss = L1_loss + 0.0001*G_img_loss + 0.01*Ez_loss
    return EG_loss, L1_loss, G_img_loss, Ez_loss


def loss_discriminator_z(netD_z, z_prior, z_enc, real_label, fake_label):
    """
    Calcul de la perte pour le discriminateur de z
    """
    Dz_loss = BCE(netD_z(z_prior), real_label) + BCE(netD_z(z_enc), fake_label)
    return Dz_loss


def loss_discriminator_img(netD_img, img_real, img_fake, real_label, fake_label):
    """
    Calcul de la perte pour le discriminateur d'images
    """
    D_loss = BCE(netD_img(img_real), real_label) + BCE(netD_img(img_fake), fake_label)
    return D_loss
