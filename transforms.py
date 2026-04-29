import torchvision.transforms as transforms
MEAN = [0.7228240966796875, 0.5555058717727661, 0.5389671325683594]
STD = [0.18855655193328857, 0.19735924899578094, 0.21101026237010956]

# La pipeline de base (sans normalisation)
transform_base = transforms.Compose([
    transforms.ToTensor(),
])

# La pipeline avec normalisation
transform_normalise = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD)
])

train_transform_aug = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD)
])

# La validation : pas d’augmentation
val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD)
])

