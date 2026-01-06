import torch
import os

# chemins
data_root = "./"        # dossier contenant 'mnist'
outf = "./results_caae"
os.makedirs(outf, exist_ok=True)

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# paramètres généraux
img_size = 32
batchSize = 64
niter = 20

# dimensions réseaux
n_channel = 1
n_encode = 64
n_gen = 64
n_disc = 64
n_z = 128
n_classes = 10

# learning rates
lr_e = 0.0002
lr_g = 0.0002
lr_d = 0.0001
