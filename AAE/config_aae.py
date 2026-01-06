import torch
import os

# Chemins
data_root = "./"          # chemin vers le dossier contenant 'mnist'
outf = "./result_aae"     # dossier pour sauvegarder les résultats
os.makedirs(outf, exist_ok=True)


#  Paramètres généraux
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
img_size = 32
batchSize = 64
niter = 20               # nombre d'epochs


# Dimensions réseaux
n_channel = 1   # MNIST = grayscale
n_encode = 64
n_gen = 64
n_disc = 64
n_z = 128       # dimension latente

# Learning rates
lr_e = 0.0002   # Encoder
lr_g = 0.0002   # Generator
lr_d = 0.0001   # Discriminateurs (img + z)
