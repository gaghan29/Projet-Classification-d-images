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

TRAIN_DIR = "melanoma-cancer-dataset/train"

# os.listdir() retourne la liste des elements dans un dossier
classes = sorted(os.listdir(TRAIN_DIR))
num_classes = len(classes)
print(f"Nombre de classes : {num_classes}")
print(f"Classes : {classes}")

# Compter les images par classe
for classe in classes:
    chemin_classe = os.path.join(TRAIN_DIR, classe)
    nb_images = len(os.listdir(chemin_classe))
    print(f" {classe} : {nb_images} images")

# Charger une image (adaptez le chemin)
chemin = os.path.join(TRAIN_DIR, "Benign", os.listdir(os.path.join(TRAIN_DIR, "Benign"))
[0])
img_pil = Image.open(chemin).convert("RGB")
print(f"Taille originale (PIL) : {img_pil.size}") # format PIL : (largeur, hauteur)
print(f"Type des pixels PIL : {type(img_pil.getpixel((0,0)))}") # tuple d’entiers 0-255

# Convertir en tenseur PyTorch
to_tensor = transforms.ToTensor()
img_tensor = to_tensor(img_pil)

print(f"Forme du tenseur : {img_tensor.shape}") # [C, H, W]
print(f"Valeur min : {img_tensor.min():.4f}")
print(f"Valeur max : {img_tensor.max():.4f}")

plt.subplot(141); plt.title("Image originale"); plt.imshow(img_tensor.permute(1, 2, 0).numpy())
plt.subplot(142); plt.title("Canal R"); plt.imshow(img_tensor[0].numpy(), cmap='Reds')
plt.subplot(143); plt.title("Canal G"); plt.imshow(img_tensor[1].numpy(), cmap='Greens')
plt.subplot(144); plt.title("Canal B"); plt.imshow(img_tensor[2].numpy(), cmap='Blues')
plt.suptitle("Canaux RGB séparés du tenseur de la forme [3, 224, 224]")
plt.show()