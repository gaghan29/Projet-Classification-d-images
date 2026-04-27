from dataset import MelonaDataset
#from transforms import transform_base, transform_normalise, MEAN, STD
from model import SimpleCNN, compter_parametres
from train import train_one_epoch, evaluate
import os
import time
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.tensorboard import SummaryWriter

from sklearn.metrics import confusion_matrix
import seaborn as sns

# --------------------------------------------------
# Configuration du device (GPU si disponible)
# --------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device utilise : {device}")

import os
import matplotlib.pyplot as plt
from PIL import Image

TRAIN_DIR = "melanoma-cancer-dataset/train"
classes = sorted(os.listdir(TRAIN_DIR))

# --- 1. Préparation des données et dictionnaire counts ---
counts = {}
for classe in classes:
    chemin_classe = os.path.join(TRAIN_DIR, classe)
    # On compte uniquement les fichiers (pour éviter de compter des dossiers cachés)
    nb_images = len([f for f in os.listdir(chemin_classe) if os.path.isfile(os.path.join(chemin_classe, f))])
    counts[classe] = nb_images

# --- 2. Affichage de la grille d'images (2 par classe) ---
num_classes = len(classes)
fig_img, axes = plt.subplots(num_classes, 2, figsize=(10, 4 * num_classes))

for i, classe in enumerate(classes):
    chemin_classe = os.path.join(TRAIN_DIR, classe)
    # On récupère les noms des fichiers images
    images_liste = os.listdir(chemin_classe)
    
    for j in range(2):
        img_path = os.path.join(chemin_classe, images_liste[j])
        img = Image.open(img_path)
        
        # Gestion du cas où il n'y aurait qu'une seule classe (axes serait 1D)
        ax = axes[i, j] if num_classes > 1 else axes[j]
        
        ax.imshow(img)
        ax.set_title(f"Label: {classe}")
        ax.axis('off')

plt.tight_layout()
plt.show()

# --- 3. Affichage du diagramme en barres ---
plt.figure(figsize=(8, 6))
plt.bar(counts.keys(), counts.values(), color=['skyblue', 'salmon'])

# Ajout des labels et du titre
plt.xlabel("Classes de mélanome")
plt.ylabel("Nombre d'images")
plt.title("Distribution des classes dans le dataset d'entraînement")

# Optionnel : ajouter le nombre exact au-dessus de chaque barre
for i, v in enumerate(counts.values()):
    plt.text(i, v + 5, str(v), ha='center', fontweight='bold')

plt.show()

to_tensor = transforms.ToTensor()
img_tensor = to_tensor(img)

print(f"Forme du tenseur : {img_tensor.shape}") # [C, H, W]
print(f"Valeur min : {img_tensor.min():.4f}")
print(f"Valeur max : {img_tensor.max():.4f}")

plt.subplot(141); plt.title("Image originale"); plt.imshow(img_tensor.permute(1, 2, 0).numpy())
plt.subplot(142); plt.title("Canal R"); plt.imshow(img_tensor[0].numpy(), cmap='Reds')
plt.subplot(143); plt.title("Canal G"); plt.imshow(img_tensor[1].numpy(), cmap='Greens')
plt.subplot(144); plt.title("Canal B"); plt.imshow(img_tensor[2].numpy(), cmap='Blues')
plt.suptitle(f"Canaux RGB séparés du tenseur de la forme : {img_tensor.shape} ")
plt.show()

transform_base = transforms.Compose([
    transforms.ToTensor(),         # Convertit les pixels (0-255) en tenseurs (0.0-1.0)
])

train_dataset = MelonaDataset("melanoma-cancer-dataset/train", transform=transform_base)
val_dataset = MelonaDataset("melanoma-cancer-dataset/test", transform=transform_base)

print(f"Taille du train set : {len(train_dataset)}")
print(f"Taille du val set : {len(val_dataset)}")
print(f"Classes : {train_dataset.classes}")
print(f"Mapping classe->entier : {train_dataset.class_to_idx}")

# Creer les DataLoaders
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=1)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=1)

# Verifier la forme d’un batch
images_batch, labels_batch = next(iter(train_loader))
print(f"Forme d’un batch d’images : {images_batch.shape}")
# Attendu : [32/64, 3, 224, 224]
print(f"Forme des labels : {labels_batch.shape}")
# Attendu : [32] ou [64]

fig, axes = plt.subplots(2, 4, figsize=(15, 8))
axes = axes.flatten()  # On aplatit la grille pour boucler facilement de 0 à 7

for i in range(8):
    # Transformation du tenseur pour l'affichage
    # .permute(1,2,0) passe de (3, 224, 224) à (224, 224, 3)
    img_display = images_batch[i].permute(1, 2, 0).numpy()
    
    # Récupération du nom de la classe
    class_name = train_dataset.classes[labels_batch[i].item()]
    
    # Affichage
    axes[i].imshow(img_display)
    axes[i].set_title(class_name)
    axes[i].axis('off')

plt.tight_layout()
plt.show()

val_images_batch, val_labels_batch = next(iter(val_loader))

fig, axes = plt.subplots(2, 4, figsize=(15, 8))
axes = axes.flatten()  # On aplatit la grille pour boucler facilement de 0 à 7

for i in range(8):
    # Transformation du tenseur pour l'affichage
    # .permute(1,2,0) passe de (3, 224, 224) à (224, 224, 3)
    val_img_display = val_images_batch[i].permute(1, 2, 0).numpy()
    
    # Récupération du nom de la classe
    class_name = val_dataset.classes[val_labels_batch[i].item()]
    
    # Affichage
    axes[i].imshow(val_img_display)
    axes[i].set_title(class_name)
    axes[i].axis('off')

plt.tight_layout()
plt.show()

model = SimpleCNN(num_classes=num_classes).to(device)

# On affiche le résultat ici
compter_parametres(model)

# SGD classique
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
# Adam (recommande pour commencer)
#optimizer = optim.Adam(model.parameters(), lr=1e-3)

NUM_EPOCHS = 20

# Critere de loss et optimiseur
criterion = nn.CrossEntropyLoss()

# Historique pour les courbes
history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

for epoch in range(1, NUM_EPOCHS + 1):
    t0 = time.time()
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer,
    device)
    val_loss, val_acc = evaluate(model, val_loader, criterion, device)
    duree = time.time() - t0

    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["train_acc"].append(train_acc)
    history["val_acc"].append(val_acc)

    print(f"Epoch {epoch:3d}/{NUM_EPOCHS} | "
        f"Loss train {train_loss:.4f} | Loss val {val_loss:.4f} | "
        f"Acc train {train_acc:.3f} | Acc val {val_acc:.3f} | "
        f"{duree:.1f}s")
    
def tracer_courbes(history, titre="CNN simple", save_name="courbes_cnn_simple.png"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    epochs = range(1, len(history["train_loss"]) + 1)

    ax1.plot(epochs, history["train_loss"], label="Train", color='steelblue')
    ax1.plot(epochs, history["val_loss"], label="Validation", color='tomato')
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.set_title(f"Loss -- {titre}"); ax1.legend(); ax1.grid(alpha=0.3)
    ax2.plot(epochs, history["train_acc"], label="Train", color='steelblue')
    ax2.plot(epochs, history["val_acc"], label="Validation", color='tomato')
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy")
    ax2.set_title(f"Accuracy -- {titre}"); ax2.legend(); ax2.grid(alpha=0.3)
    ax2.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(save_name, dpi=150)
    plt.show()

tracer_courbes(history, titre="CNN simple")