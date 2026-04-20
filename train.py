import os

TRAIN_DIR = "melanoma-cancer-dataset/train"
TEST_DIR = "melanoma-cancer-dataset/test"
# os.listdir() retourne la liste des elements dans un dossier
classes = sorted(os.listdir(TRAIN_DIR))
#classes = sorted(os.listdir(TEST_DIR))
num_classes = len(classes)
#print(f"Nombre de classes : {num_classes}")
#print(f"Classes : {classes}")

# Compter les images par classe
for classe in classes:
    chemin_classe = os.path.join(TRAIN_DIR, classe)
    #chemin_classe = os.path.join(TEST_DIR, classe)
    nb_images = len(os.listdir(chemin_classe))
    print(f" {classe} : {nb_images} images")