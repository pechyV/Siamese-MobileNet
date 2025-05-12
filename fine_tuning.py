import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import os
from modules.dataset import ChangeDetectionDataset
from modules.utils import setup_logging, load_checkpoint, save_final_model
from model.siamese_unet import get_model
from train import train
import torchvision.transforms as transforms

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

# === Nastavení Fine-Tuningu ===
learning_rate = 2e-5    # jemnější krok než 5e-5
weight_decay = 1e-6
batch_size = 16
patience = 5
min_delta = 0.0001
additional_epochs = 20  # navíc po původním tréninku
checkpoint_path = "./checkpoints_gridsearch/lr5e-05_wd1e-06_bs16/checkpoint_epoch_68.pth"  # nejlepší dosažený checkpoint
checkpoint_dir = "./checkpoints_finetune"

# Loss funkce
criterion = nn.BCEWithLogitsLoss()

# === Inicializace modelu ===
model = get_model(0)
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# Načtení posledního nejlepšího checkpointu
start_epoch = load_checkpoint(model, optimizer, checkpoint_path)
model.to(device)

# Scheduler na snížení LR při stagnaci
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5, verbose=True)

# Dataloadery
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# === Fine-tuning trénink ===
val_loss = train(
    load_pretrain=False,
    model=model,
    train_dataloader=train_dataloader,
    val_dataloader=val_dataloader,
    criterion=criterion,
    optimizer=optimizer,
    device=device,
    num_epochs=start_epoch + additional_epochs,
    start_epoch=start_epoch,
    checkpoint_dir=checkpoint_dir,
    patience=patience,
    min_delta=min_delta,
    scheduler=scheduler  # >>> Pass scheduler do train()
)

# === Uložit finální doladěný model ===
save_final_model(model, "./finetuned_best_model.pth")

print("\nFine-tuning hotov! Finální model uložen jako 'finetuned_best_model.pth'.")
