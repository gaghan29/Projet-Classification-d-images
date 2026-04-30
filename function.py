from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch
import matplotlib.pyplot as plt
import time
from train import train_one_epoch, evaluate

MEAN = [0.7228240966796875, 0.5555058717727661, 0.5389671325683594]
STD = [0.18855655193328857, 0.19735924899578094, 0.21101026237010956]

def calculer_mean_std(dataset):
    """Calcule la moyenne et l’ecart-type par canal sur le dataset."""
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    n_batches = 0
    for images, _ in loader:
        # images : [batch, 3, H, W]
        # On moyenne sur les dimensions batch, H, W (dim 0, 2, 3)
        mean += images.mean(dim=[0, 2, 3])
        std += images.std(dim=[0, 2, 3])
        n_batches += 1
    mean /= n_batches
    std /= n_batches
    print(f"Mean : {mean}")
    print(f"Std : {std}")
    return mean.tolist(), std.tolist()

def tracer_courbes(history, titre="CNN simple", save_name="courbes_cnn_simple.png"):
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        ax1, ax2, ax3, ax4 = axes.flatten()

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
        ax3.plot(epochs, history["lrs"], label="Learning rate", color='steelblue')
        ax3.set_yscale('log')
        ax3.set_xlabel("Epoch"); ax3.set_ylabel("Learning rate")
        ax3.set_title(f"Learning rate -- {titre}"); ax3.legend(); ax3.grid(alpha=0.3)
        ax4.axis('off')
        plt.tight_layout()
        plt.savefig(save_name, dpi=150)
        plt.show()


def entrainement(NUM_EPOCHS, optimizer, model, train_loader, val_loader, criterion, device):
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
    tracer_courbes(history, titre="CNN simple")

# Fonction pour dé-normaliser (remettre en [0, 1])
def denormalize(tensor, mean, std):
    # clone() pour ne pas modifier l'original
    t = tensor.clone()
    for c in range(3): # Pour chaque canal R, G, B
        t[c] = t[c] * std[c] + mean[c]
    return t

def comparer_images(I1, I2):
     # 2 & 3. Affichage côte à côte
    plt.figure(figsize=(12, 5))

    # Image Base
    plt.subplot(1, 2, 1)
    plt.imshow(I1.permute(1, 2, 0))
    plt.title(f"Originale\nMin: {I1.min():.2f}, Max: {I1.max():.2f}")
    plt.axis('off')

    I2_restored = denormalize(I2, MEAN, STD)
    # Image Normalisée (restaurée pour affichage)
    plt.subplot(1, 2, 2)
    plt.imshow(I2_restored.permute(1, 2, 0).clamp(0, 1)) # clamp par sécurité
    plt.title(f"Normalisée (Visualisation)\nMin: {I2.min():.2f}, Max: {I2.max():.2f}")
    plt.axis('off')

    plt.show()

def entrainement_sched(NUM_EPOCHS, model, train_loader, val_loader, criterion, device):
    # Historique pour les courbes
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "lrs": []}

    optimizer_sched = optim.Adam(model.parameters(), lr=1e-3)

    # Diviser le lr par 10 toutes les 7 epochs
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer_sched, step_size=7, gamma=0.1)

    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer_sched,
        device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        duree = time.time() - t0

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["lrs"].append(optimizer_sched.param_groups[0]['lr'])

        scheduler.step() # IMPORTANT : apres chaque epoch

        print(f"Epoch {epoch:3d}/{NUM_EPOCHS} | "
            f"Loss train {train_loss:.4f} | Loss val {val_loss:.4f} | "
            f"Acc train {train_acc:.3f} | Acc val {val_acc:.3f} | "
            f"Learning rate {optimizer_sched.param_groups[0]['lr']:.5f} |"
            f"{duree:.1f}s")
    tracer_courbes(history, titre="CNN simple")