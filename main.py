#from dataset import MelonaDataset
#from transforms import transform_base, transform_normalise, MEAN, STD
#from model import SimpleCNN, compter_parametres
#from train import train_one_epoch, evaluate

import torch

import os
import matplotlib.pyplot as plt
from PIL import Image
from train import TRAIN_DIR, classes
import torchvision.transforms as transforms
#import numpy as np

#Varibles importées depuis train.py
#TRAIN_DIR = "melanoma-cancer-dataset/train"
#classes = sorted(os.listdir(TRAIN_DIR)) 

# Préparation de la figure
nb_lignes = len(classes)
nb_colonnes = 2
fig, axes = plt.subplots(nb_lignes, nb_colonnes, figsize=(10, 5 * len(classes)))
axes = axes.flatten() 

idx = 0
for i in classes:
    chemin_classe = os.path.join(TRAIN_DIR, i)
    # On récupère la liste des noms de fichiers images
    images_liste = os.listdir(chemin_classe)
    
    # On prend les 2 premières images
    for i in range(2):
        img_name = images_liste[i]
        img_path = os.path.join(chemin_classe, img_name)
        
        # Charger l'image avec PIL
        img = Image.open(img_path)
        
        # Affichage dans la grille
        axes[idx].imshow(img)
        axes[idx].set_title(f"Classe : {i}")
        axes[idx].axis("off")  # Enlève les graduations x et y
        
        idx += 1

plt.suptitle("Aperçu du Dataset Melona (2 images par classe)", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Ajuste l'espacement pour éviter les chevauchements
plt.show()

# Partie 2.3 : Diagramme en barres de la distribution

# Construction du dictionnaire counts
counts = {}
for classe in classes:
    chemin_classe = os.path.join(TRAIN_DIR, classe)
    nb_images = len(os.listdir(chemin_classe))
    counts[classe] = nb_images

# Affichage du diagramme en barres
plt.figure(figsize=(10, 6))
plt.bar(counts.keys(), counts.values(), color=['skyblue', 'salmon'])

# Personnalisation du graphique
plt.xlabel("Classes de mélanome")
plt.ylabel("Nombre d'images")
plt.title("Distribution des images par classe (Dataset d'entraînement)")

# Ajout du nombre exact d'images au-dessus de chaque barre
for i, v in enumerate(counts.values()):
    plt.text(i, v + 5, str(v), ha='center', fontweight='bold')

plt.show()

# Part 3.1 : Inspecter une image

# Charger une image (adaptez le chemin)
chemin = os.path.join(TRAIN_DIR, "Benign", os.listdir(os.path.join(TRAIN_DIR, "Benign"))[50])
img_pil = Image.open(chemin).convert("RGB")
print(f"Taille originale (PIL) : {img_pil.size}") # format PIL : (largeur, hauteur)
print(f"Type des pixels PIL : {type(img_pil.getpixel((0,0)))}") # tuple d’entiers 0-255

# Convertir en tenseur PyTorch
to_tensor = transforms.ToTensor()
img_tensor = to_tensor(img_pil)

print(f"Forme du tenseur : {img_tensor.shape}") # [C, H, W]
print(f"Valeur min : {img_tensor.min():.4f}")
print(f"Valeur max : {img_tensor.max():.4f}")




# Partie 3.2. Visualiser les canaux RGB séparément
fig, axes = plt.subplots(1, 4, figsize=(15, 5))

axes[0].imshow(img_tensor.permute(1, 2, 0).numpy())#, cmap='Reds')
axes[0].set_title("Image Originale")
axes[0].axis("off")

# 3. Affichage des canaux R, G et B séparément
canaux = ["Rouge (R)", "Vert (G)", "Bleu (B)"]
cmaps = ["Reds", "Greens", "Blues"]

for j in range(3):
    axes[j+1].imshow(img_tensor[j].numpy(), cmap=cmaps[j])
    axes[j+1].set_title(canaux[j])
    axes[j+1].axis("off") 

plt.show()

# Affichage des dimensions du tenseur d'une image d'un grain de beauté
plt.suptitle(img_tensor)
plt.show()

#plt.suptitle(f"Visualisation des canaux - Forme du tenseur : {img_tensor.shape}", fontsize=16)
#plt.show()