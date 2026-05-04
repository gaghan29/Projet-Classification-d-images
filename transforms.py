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
    transforms.RandomResizedCrop(size=224, scale=(0.8, 1.0)), 
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
    transforms.RandomErasing(p=0.15)
])

# La validation : pas d’augmentation
val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD)
])

