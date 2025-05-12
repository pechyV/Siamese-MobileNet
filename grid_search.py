import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from itertools import product
import os
from modules.dataset import ChangeDetectionDataset
from modules.utils import setup_logging, load_checkpoint, save_final_model
from model.siamese_unet import get_model
from train import train
import torchvision.transforms as transforms
import torch.nn as nn

setup_logging()

# Dataset a transformace
train_root_dir = "../dataset/train/"
val_root_dir = "../dataset/val/"
transform = transforms.Compose([
    transforms.ToTensor()
])
train_dataset = ChangeDetectionDataset(train_root_dir, transform=transform, augment=True)
val_dataset = ChangeDetectionDataset(val_root_dir, transform=transform, augment=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Definice grid search parametru
learning_rates = [1e-4, 5e-5, 1e-5]
weight_decays = [1e-5, 1e-6, 1e-4]
batch_sizes = [16]  # Necháme batch_size konstantní (nebo přidej více hodnot)

# Cesta k checkpointu
checkpoint_path = "./checkpoints/checkpoint_epoch_9.pth"  # Nastav správnou cestu
checkpoint_dir = "./checkpoints_gridsearch"

# Training parametry
criterion = nn.BCEWithLogitsLoss()
num_epochs = 15  # Počet epoch pro každou kombinaci
patience = 5
min_delta = 0.0001

best_val_loss = float('inf')
best_model_info = {}

for lr, wd, batch_size in product(learning_rates, weight_decays, batch_sizes):
    print(f"\n==> Spouštím grid search kombinaci: lr={lr}, wd={wd}, bs={batch_size}")

    # Speciální checkpoint adresář pro tuto kombinaci
    exp_checkpoint_dir = os.path.join(checkpoint_dir, f"lr{lr}_wd{wd}_bs{batch_size}")

    # Přeskoč kombinaci, pokud složka už existuje a obsahuje checkpointy
    if os.path.exists(exp_checkpoint_dir) and any(fname.endswith(".pth") for fname in os.listdir(exp_checkpoint_dir)):
        print(f"[SKIP] Kombinace už byla zřejmě trénována: LR={lr}, WD={wd}, BS={batch_size}")
        continue

    # Vytvoření složky, pokud neexistuje
    os.makedirs(exp_checkpoint_dir, exist_ok=True)

    model = get_model(0)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    start_epoch = load_checkpoint(model, optimizer, checkpoint_path)
    model.to(device)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Vrať validační loss
    val_loss = train(
        load_pretrain=False, model=model, 
        train_dataloader=train_dataloader, 
        val_dataloader=val_dataloader, 
        criterion=criterion, optimizer=optimizer,
        device=device, num_epochs=start_epoch + num_epochs,
        start_epoch=start_epoch,
        checkpoint_dir=exp_checkpoint_dir,
        patience=patience, 
        min_delta=min_delta
    )

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_info = {
            "lr": lr,
            "wd": wd,
            "bs": batch_size
        }
        # Uložit nejlepší model
        save_final_model(model, f"./best_model.pth")

print("\n=== Grid Search hotov ===")
print(f"Nejlepší kombinace: {best_model_info} s Validation Loss: {best_val_loss:.6f}")
