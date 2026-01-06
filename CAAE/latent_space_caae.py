import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np

def visualize_latent_space(embeddings, labels):
    # PCA 2D
    pca2d = PCA(n_components=2)
    z_pca2d = pca2d.fit_transform(embeddings)
    plt.figure(figsize=(10,8))
    plt.scatter(z_pca2d[:,0], z_pca2d[:,1], c=labels, cmap='tab20', alpha=0.7)
    plt.colorbar(label='Classe MNIST')
    plt.title("Espace latent - PCA 2D")
    plt.show()

    # PCA 3D
    pca3d = PCA(n_components=3)
    z_pca3d = pca3d.fit_transform(embeddings)
    fig = plt.figure(figsize=(12,10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(z_pca3d[:,0], z_pca3d[:,1], z_pca3d[:,2], c=labels, cmap='tab20', alpha=0.7)
    ax.set_title("Espace latent - PCA 3D")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_zlabel("PC3")
    plt.show()

    # t-SNE 2D
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    z_tsne = tsne.fit_transform(embeddings)
    plt.figure(figsize=(12,10))
    scatter = plt.scatter(z_tsne[:,0], z_tsne[:,1], c=labels, cmap='tab10', alpha=0.6, s=20)
    plt.colorbar(scatter, label='Chiffre')
    plt.title("Espace latent - t-SNE 2D")
    plt.xlabel("Dimension 1"); plt.ylabel("Dimension 2")
    plt.tight_layout()
    plt.show()
    
    # TSNE 3D
    tsne3 = TSNE(n_components=3, perplexity=30, n_iter=1000, random_state=42)
    z_tsne3 = tsne3.fit_transform(embeddings)

    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111, projection='3d')
    p = ax.scatter(z_tsne3[:,0], z_tsne3[:,1], z_tsne3[:,2],
               c=labels_list, cmap='tab10', alpha=0.7, s=20)
    ax.set_title("Espace latent du CAAE - t-SNE 3D")
    ax.set_xlabel("t-SNE1")
    ax.set_ylabel("t-SNE2")
    ax.set_zlabel("t-SNE3")

    # Légende pour classes
    handles = [plt.Line2D([0],[0], marker='o', color='w', markerfacecolor=plt.cm.tab10(i/10), markersize=6) for i in range(10)]
    ax.legend(handles, [str(i) for i in range(10)], title="Classes")
    plt.show()
