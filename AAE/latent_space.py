import torch
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def plot_latent_space(netE, dataloader, device):
    netE.eval()
    embeddings = []
    labels_list = []

    with torch.no_grad():
        for img, lbl in dataloader:
            img = img.to(device)
            z = netE(img)
            embeddings.append(z.cpu().numpy())
            labels_list.append(lbl.numpy())

    embeddings = np.concatenate(embeddings, axis=0)
    labels_list = np.concatenate(labels_list, axis=0)

    # PCA 2D
    pca2d = PCA(n_components=2)
    z_pca2d = pca2d.fit_transform(embeddings)
    plt.figure(figsize=(10,8))
    plt.scatter(z_pca2d[:,0], z_pca2d[:,1], c=labels_list, cmap='tab10', alpha=0.6, s=20)
    plt.colorbar(label='Chiffre')
    plt.title("Espace latent PCA 2D")
    plt.show()

    # PCA 3D
    pca3d = PCA(n_components=3)
    z_pca3d = pca3d.fit_transform(embeddings)

    fig = plt.figure(figsize=(12,10))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(z_pca3d[:,0], z_pca3d[:,1], z_pca3d[:,2], c=labels_list, cmap='tab10', alpha=0.6, s=20)
    fig.colorbar(scatter, ax=ax, label='Chiffre')
    ax.set_title("Espace latent AAE non-conditionnel - PCA 3D")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    plt.show()

    # t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    z_tsne = tsne.fit_transform(embeddings)
    plt.figure(figsize=(12,10))
    plt.scatter(z_tsne[:,0], z_tsne[:,1], c=labels_list, cmap='tab10', alpha=0.6, s=20)
    plt.colorbar(label='Chiffre')
    plt.title("Espace latent t-SNE")
    plt.show()
