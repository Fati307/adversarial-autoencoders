import torch
import torch.nn as nn
import torch.nn.functional as F

BCE = nn.BCELoss()
CE  = nn.CrossEntropyLoss()
MSE = nn.MSELoss()

# KL Divergence
def kl_divergence(mu, logvar):
    """
    KL divergence between N(mu, sigma) and N(0,1)
    """
    return -0.5 * torch.mean(
        1 + logvar - mu.pow(2) - logvar.exp()
    )
