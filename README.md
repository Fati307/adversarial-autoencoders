# adversarial-autoencoders

Ce dépôt contient le code source de trois implémentations d'Adversarial Autoencoders pour l'apprentissage de représentations latentes sur MNIST, développées avec PyTorch.

## Table des matières

- [Prérequis](#prérequis)
- [Démarrage rapide](#démarrage-rapide)
  - [Cloner le dépôt](#cloner-le-dépôt)
  - [Configuration de l'environnement](#configuration-de-lenvironnement)
  - [Préparation des données](#préparation-des-données)
  - [Exécution de l'entraînement](#exécution-de-lentraînement)
- [Architecture des Modèles](#architecture-des-modèles)
- [Visualisation et Résultats](#visualisation-et-résultats)
- [📄 Document de Référence Technique](#-document-de-référence-technique)

---

## 1. Prérequis

Avant de commencer, assurez-vous d'avoir les éléments suivants installés et configurés :

- **Python 3.8** ou supérieur
- **PyTorch 2.0+** avec support CUDA (recommandé)
- **pip** ou **conda** pour la gestion des packages
- **Git** pour cloner le dépôt
- **GPU NVIDIA** (optionnel mais fortement recommandé pour l'entraînement)

**Packages Python requis :**
- `torch>=2.0.0`
- `torchvision>=0.15.0`
- `numpy>=1.24.0`
- `matplotlib>=3.7.0`
- `scikit-learn>=1.3.0`
- `tqdm>=4.65.0`

---

## 2. Démarrage rapide

### Cloner le dépôt

**Avec SSH :**
```bash
git clone git@github.com:Fati307/adversarial-autoencoders.git
cd adversarial-autoencoders
```

**Avec HTTPS :**
```bash
git clone https://github.com/Fati307/adversarial-autoencoders.git
cd adversarial-autoencoders
```

### Configuration de l'environnement

Installez les dépendances depuis la racine du projet :

```bash
pip install -r requirements.txt
```

**Contenu du fichier `requirements.txt` :**
```
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
matplotlib>=3.7.0
scikit-learn>=1.3.0
tqdm>=4.65.0
```

### Préparation des données

Le projet utilise le dataset MNIST organisé en structure `ImageFolder`. Placez vos données dans le dossier suivant :

```
data/
└── mnist/
    ├── 0/
    │   ├── img1.png
    │   ├── img2.png
    │   └── ...
    ├── 1/
    │   ├── img1.png
    │   └── ...
    └── ...
```

Le fichier `dataset.py` à la racine du projet gère automatiquement le chargement des données pour tous les modèles.

### Exécution de l'entraînement

**Pour AAE (Adversarial Autoencoder) :**
```bash
cd AAE
python train_aae.py
```

**Pour CAAE (Conditional Adversarial Autoencoder) :**
```bash
cd CAAE
python train_caae.py
```

**Pour Hybrid CAAE :**
```bash
cd CAAEHybrid
python train_h_caae.py
```

Les résultats (images de reconstruction, visualisations du latent space) seront sauvegardés dans les dossiers `result_aae/`, `result_caae/`, et `result_h_caae/`.

---

## 3. Architecture des Modèles

### Structure du projet

```
adversarial-autoencoders/
│
├── dataset.py                     # Dataset commun pour les 3 modèles
├── requirements.txt               # Dépendances Python
├── README.md                      # Documentation du projet
│
├── AAE/                           # Adversarial Autoencoder (Non-conditionnel)
│   ├── config_aae.py              # Hyperparamètres AAE
│   ├── models/
│   │   ├── encoder.py             # Encodeur CNN
│   │   ├── decoder.py             # Décodeur CNN transposé
│   │   ├── discriminator_img.py  # Discriminateur d'images
│   │   └── discriminator_z.py    # Discriminateur latent
│   ├── losses_aae.py              # Fonctions de perte
│   ├── train_aae.py               # Script d'entraînement
│   └── utils_aae.py               # Fonctions utilitaires (affichage, init)
│
├── CAAE/                          # Conditional Adversarial Autoencoder
│   ├── config_caae.py             # Hyperparamètres CAAE
│   ├── models/
│   │   ├── encoder.py             # Encodeur VAE-style avec reparamétrisation
│   │   ├── decoder.py             # Décodeur conditionnel (z + classe)
│   │   ├── discriminator_img.py  # Discriminateur d'images
│   │   └── discriminator_z.py    # Discriminateur latent
│   ├── losses_caae.py             # Perte reconstruction + KL + adversarial + classification
│   ├── train_caae.py              # Script d'entraînement avec conditionnement
│   └── utils_caae.py              # One-hot encoding & visualisation
│
└── CAAEHybrid/                    # Hybrid Conditional AAE
    ├── config_h_caae.py           # Hyperparamètres Hybrid
    ├── models/
    │   ├── encoder.py             # Encodeur conditionnel (image + classe → z)
    │   ├── decoder.py             # Décodeur conditionnel (z + classe → image)
    │   ├── discriminator_img.py  # Double tête: real/fake + classification
    │   └── discriminator_z.py    # Discriminateur latent conditionnel
    ├── losses_h_caae.py           # Perte hybride: L1 + adversarial + classification
    ├── train_h_caae.py            # Entraînement multi-objectif
    └── utils_h_caae.py            # Class maps & utilitaires avancés
```

### Comparaison des modèles

| Caractéristique | AAE | CAAE | Hybrid CAAE |
|-----------------|-----|------|-------------|
| **Type** | Non-conditionnel | Conditionnel VAE | Conditionnel Hybride |
| **Conditionnement** | ❌ | ✅ Labels de classe | ✅ Labels de classe |
| **Régularisation latente** | Adversarial | KL + Adversarial | Adversarial |
| **Discriminateur d'images** | ✅ | ✅ | ✅ + Classification auxiliaire |
| **Discriminateur latent** | ✅ | ✅ | ✅ Conditionnel |
| **Génération contrôlée** | Aléatoire | Par classe | Par classe |
| **Complexité** | Faible | Moyenne | Élevée |
| **Cas d'usage** | Exploration non-supervisée | Génération contrôlée | Applications avancées |

---

## 4. Visualisation et Résultats

Chaque script d'entraînement génère automatiquement :

1. **Reconstructions d'images** : Comparaison originales vs reconstructions à chaque epoch
2. **Visualisation du latent space** :
   - Projection PCA 2D et 3D
   - Projection t-SNE 2D et 3D
3. **Génération conditionnelle** (CAAE et Hybrid) : Génération de chiffres spécifiques par classe

**Exemple d'utilisation pour générer des images d'un chiffre spécifique :**

```python
from CAAE.models.decoder import Generator
from CAAE.utils_caae import one_hot_label_tensor
import torch

# Charger le modèle
netG = Generator().to(device)
netG.load_state_dict(torch.load('checkpoint_gen.pth'))
netG.eval()

# Générer 16 images du chiffre 7
z = torch.randn(16, 128, device=device)
labels = torch.full((16,), 7, device=device)
class_ohe = one_hot_label_tensor(labels, 10, device)

with torch.no_grad():
    images = netG(z, class_ohe)
```

---

## 5. 📄 Document de Référence Technique

### 👩‍💻 Présentation

Ce document décrit l'architecture des trois variantes d'Adversarial Autoencoders implémentées pour l'apprentissage de représentations latentes robustes sur MNIST.  
Il définit les bonnes pratiques de développement, l'organisation du code, la configuration des hyperparamètres, ainsi que les règles à respecter pour contribuer au projet.

### ✅ Objectifs

- Avoir un cadre clair et homogène pour le développement des modèles génératifs
- Assurer la lisibilité, maintenabilité et reproductibilité du code
- Faciliter l'expérimentation avec différentes architectures
- Optimiser la collaboration et le partage des résultats

### 🧱 Structure du Code

Chaque modèle (AAE, CAAE, Hybrid CAAE) suit la même organisation :

```
ModelName/
├── config_*.py          # Tous les hyperparamètres centralisés
├── models/              # Architectures des réseaux
│   ├── encoder.py
│   ├── decoder.py
│   ├── discriminator_img.py
│   └── discriminator_z.py
├── losses_*.py          # Définition des fonctions de perte
├── train_*.py           # Boucle d'entraînement principale
└── utils_*.py           # Fonctions utilitaires (affichage, one-hot, init)
```

**Règles d'organisation :**

- **Séparation des préoccupations** : Chaque fichier a une responsabilité unique
- **Modularité** : Les modèles peuvent être importés et réutilisés facilement
- **Configuration centralisée** : Tous les hyperparamètres dans `config_*.py`
- **Pas de duplication** : Le `dataset.py` commun est partagé par tous les modèles

### 🧑‍💻 Règles de Développement

| Règle | Description |
|-------|-------------|
| ✍️ **Langue** | Anglais pour le code (classes, variables, commentaires). Français accepté pour la documentation. |
| 🧠 **Clarté** | Bien comprendre l'architecture avant de modifier le code. |
| 📝 **Commentaires** | Commentez les parties non-évidentes, surtout dans les losses et les architectures. |
| 🧩 **Nomination** | Respectez les conventions Python : `snake_case` pour variables/fonctions, `PascalCase` pour classes. |
| 🔁 **Expérimentation** | Documentez vos expérimentations (hyperparamètres testés, résultats) dans un fichier `experiments.md`. |
| 🤝 **Reproductibilité** | Fixez les seeds aléatoires (`torch.manual_seed()`) pour garantir la reproductibilité. |

### ⚙️ Fichiers de Configuration

#### config_*.py

Chaque modèle possède son propre fichier de configuration. Exemple pour AAE :

```python
# config_aae.py
import torch

# Architecture
n_channel = 1        # MNIST = grayscale
n_disc = 64          # Canaux discriminateur
n_gen = 64           # Canaux générateur
n_encode = 64        # Canaux encodeur
n_z = 128            # Dimension latente

# Training
img_size = 32
batchSize = 64
niter = 20
lr_e = 0.0002        # Learning rate encodeur
lr_g = 0.0002        # Learning rate générateur
lr_d = 0.0001        # Learning rate discriminateurs

# Environnement
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
outf = './result_aae'
data_root = './data'
```

**⚠️ Important :**
- Ne jamais hardcoder des chemins absolus
- Utiliser `os.path.join()` pour la compatibilité multi-plateforme
- Documenter chaque hyperparamètre

### 🔬 Fonctions de Perte

#### AAE
```python
L_total = L1(x_recon, x_original) 
          + λ_img * BCE(D_img(x_recon), 1) 
          + λ_z * BCE(D_z(z), 1)
```

#### CAAE
```python
L_total = MSE(x_recon, x_original)
          + λ_adv * BCE(D_img(x_gen), 1)
          + λ_z * BCE(D_z(z), 1)
          + CrossEntropy(y_pred, y_true)
          + β * KL(q(z|x) || p(z))
```

#### Hybrid CAAE
```python
L_total = L1(x_recon, x_original)
          + λ_img * BCE(D_img(x_recon), 1)
          + λ_z * BCE(D_z(z|y), 1)
          + CrossEntropy(y_pred, y_true)
```

### 🧪 Bonnes Pratiques Supplémentaires

1. **Initialisation des poids** : Utiliser `weights_init()` avec `nn.init.normal_()` pour stabilité
2. **Label smoothing** : Utiliser 0.9 pour real, 0.1 pour fake (stabilise l'entraînement GAN)
3. **Gradient clipping** : Éviter les explosions de gradients
4. **Monitoring** : Logger toutes les losses séparément avec `tqdm`
5. **Checkpointing** : Sauvegarder les modèles régulièrement avec `torch.save()`
6. **Visualisation** : Afficher les reconstructions à chaque epoch pour détecter les problèmes
7. **Memory management** : Appeler `torch.cuda.empty_cache()` et `gc.collect()` après chaque epoch

### 📊 Métriques d'Évaluation

Pour chaque modèle, suivez ces métriques :

- **Loss de reconstruction** : L1 ou MSE entre images originales et reconstruites
- **Loss adversarial** : Performance des discriminateurs (D_img et D_z)
- **Accuracy de classification** (CAAE/Hybrid) : Précision sur les labels prédits
- **KL divergence** (CAAE) : Distance entre distribution encodée et prior
- **Qualité visuelle** : Inspection manuelle des reconstructions et générations

### 🔄 Workflow de Contribution

1. **Créer une branche** pour votre expérimentation :
   ```bash
   git checkout -b experiment/nouveau-modele
   ```

2. **Développer** en suivant la structure existante

3. **Tester** votre code avec un petit nombre d'epochs

4. **Documenter** vos résultats dans `experiments.md`

5. **Commit** avec des messages clairs :
   ```bash
   git commit -m "feat(CAAE): Add KL annealing for better convergence"
   ```

6. **Push** et créer une Pull Request :
   ```bash
   git push origin experiment/nouveau-modele
   ```

### 📚 Ressources et Références

**Papers fondateurs :**
- Makhzani et al. (2015) - Adversarial Autoencoders
- Kingma & Welling (2013) - Auto-Encoding Variational Bayes
- Goodfellow et al. (2014) - Generative Adversarial Networks

**Documentation PyTorch :**
- [torch.nn](https://pytorch.org/docs/stable/nn.html)
- [torch.optim](https://pytorch.org/docs/stable/optim.html)
- [torchvision.transforms](https://pytorch.org/vision/stable/transforms.html)

### 📥 Besoin d'aide ?

Pour toute question sur :
- L'architecture des modèles
- Les hyperparamètres optimaux
- Les problèmes d'entraînement
- L'ajout de nouvelles fonctionnalités

Ouvrez une issue sur GitHub ou contactez les mainteneurs du projet.

---

**Développé avec ❤️ et PyTorch**