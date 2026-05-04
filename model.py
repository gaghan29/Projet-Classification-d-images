import torch.nn as nn
from torchvision import models

class SimpleCNN(nn.Module):

    def __init__(self, num_classes):
        super().__init__()

        self.bloc1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.bloc2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.bloc3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classificateur = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 28 * 28, 256), # 224 -> 112 -> 56 -> 28
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.bloc1(x)
        x = self.bloc2(x)
        x = self.bloc3(x)
        x = self.classificateur(x)
        return x

class ResNet(nn.Module):
    def __init__(self, num_classes, freeze_backbone=True):
        super().__init__()
        
        # 1. Charger le moteur (backbone)
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        
        # 2. Geler les paramètres si demandé
        for param in self.resnet.parameters():
            param.requires_grad = False
        
        # 3. Remplacer la tête de classification (resnet.fc)
        # On récupère le nombre de neurones en entrée de la couche finale (512 pour ResNet18)
        in_features = self.resnet.fc.in_features
        
        # On remplace par ta propre structure (on peut même remettre un petit Sequential)
        self.resnet.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)

def compter_parametres(model):
    entrainables = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parametres entrainables : {entrainables:,}")

class ResNet_FT(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # 1. Charger ResNet18
        self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        
        # 2. On commence par tout geler
        for param in self.model.parameters():
            param.requires_grad = False
            
        # 3. Remplacer la tête fc (elle est True par défaut à la création)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)
        
        # 4. Dégeler spécifiquement la layer4
        for param in self.model.layer4.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.model(x)

    def get_optimizer_params(self):
        return [
            {"params": self.model.layer4.parameters(), "lr": 1e-4},
            {"params": self.model.fc.parameters(), "lr": 1e-3},
        ]