import os
from PIL import Image
from torch.utils.data import Dataset

class MelonaDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Chemin vers le dossier (train ou test).
            transform (callable, optional): Transformations à appliquer sur les images.
        """
        self.root_dir = root_dir
        self.transform = transform
        
        # On récupère la liste des classes (noms des sous-dossiers)
        self.classes = sorted(os.listdir(root_dir))
        
        # On crée une liste de tuples (chemin_image, index_classe)
        self.data = []
        for idx, cls_name in enumerate(self.classes):
            cls_path = os.path.join(root_dir, cls_name)
            for img_name in os.listdir(cls_path):
                img_path = os.path.join(cls_path, img_name)
                self.data.append((img_path, idx))

    def __len__(self):
        # Retourne le nombre total d'images
        return len(self.data)

    def __getitem__(self, idx):
        # Charge une image et son label par son index
        img_path, label = self.data[idx]
        image = Image.open(img_path).convert("RGB") # On force en RGB
        
        if self.transform:
            image = self.transform(image)
            
        return image, label