import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from modules.dataset import ChangeDetectionDataset
from model.siamese_unet import get_model
from modules.utils import visualize_results, setup_logging, save_checkpoint, load_checkpoint, save_final_model, load_pretrained_model
from modules.early_stop import EarlyStopping
import logging
import os
import random

# Nastavení logování
setup_logging()

def train(load_pretrain, model, train_dataloader, val_dataloader, criterion, optimizer, device, num_epochs, checkpoint_dir="./checkpoints/", patience = 5, min_delta = 0.0001):

    start_epoch = 0 # set checkpoint > 0
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{start_epoch}.pth")
    
    if not load_pretrain and start_epoch > 0 and os.path.exists(checkpoint_path):
        start_epoch = load_checkpoint(model, optimizer, checkpoint_path)
        model.to(device)
        logging.info(f"Continue from epoch {start_epoch+1}")
    
    early_stopping = EarlyStopping(patience, min_delta)

    for epoch in range(start_epoch, num_epochs):
        model.train()  # Přepnutí modelu do režimu trénování
        epoch_loss = 0.0
        for t1, t2, mask in train_dataloader:
            t1, t2, mask = t1.to(device), t2.to(device), mask.to(device)
            optimizer.zero_grad()
            outputs = model(t1, t2)
            loss = criterion(outputs, mask)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        logging.info(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {epoch_loss/len(train_dataloader):.4f}")

        # Validace po každé epoše
        model.eval()  # Přepnutí modelu do režimu evaluace
        val_loss = 0.0
        with torch.no_grad():
            for t1, t2, mask in val_dataloader:
                t1, t2, mask = t1.to(device), t2.to(device), mask.to(device)
                outputs = model(t1, t2)
                loss = criterion(outputs, mask)
                val_loss += loss.item()

        logging.info(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss/len(val_dataloader):.4f}")

        # Zavolání EarlyStopping
        early_stopping(val_loss / len(val_dataloader))  # Předání průměrné validační ztráty

        if early_stopping.early_stop:
            logging.info("EARLY STOPPING: training shuted down.")
            break

        # Uložení checkpointu po každé epoše
        save_checkpoint(model, optimizer, epoch, checkpoint_dir)
        
        # Vizualizace výsledků po každé epoše
        with torch.no_grad():
            indices = random.sample(range(len(t1)), min(5, len(t1)))  # Unikátní indexy
            for idx in indices:
                t1_sample, t2_sample, mask_sample = t1[idx], t2[idx], mask[idx]
                pred_sample = (outputs > 0.5).float()[idx]

                save_visualization(t1_sample, t2_sample, mask_sample, pred_sample, epoch+1, idx)


# Funkce pro ukládání výsledků
def save_visualization(t1_sample, t2_sample, mask_sample, pred_sample, epoch, idx):
    save_dir = "./visualizations"
    epoch_dir = os.path.join(save_dir, f"epoch_{epoch}")
    os.makedirs(epoch_dir, exist_ok=True)
    
    visualize_results(t1_sample, t2_sample, mask_sample, pred_sample, epoch, idx, save_dir)


if __name__ == "__main__":
    """ Parametry """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(0)
    load_pretrain = False
    learning_rate = 0.0001
    num_epochs = 100
    batch_size = 16
    patience = 10
    min_delta = 0.0001
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), learning_rate, weight_decay=1e-5)
    #optimizer = torch.optim.Adam(model.parameters(), learning_rate, weight_decay=1e-5)
    train_root_dir = "./test_dataset/train/"
    val_root_dir = "./test_dataset/val/"
    out_model = "./trained_model/siamese_unet.pth"
    pretrained_model = ""
    checkpoint_dir = "./checkpoints"
    
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    train_dataset = ChangeDetectionDataset(train_root_dir, transform=transform, augment=True)
    val_dataset = ChangeDetectionDataset(val_root_dir, transform=transform, augment=False)


    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size, shuffle=False)

    # Načtení předtrénovaného modelu
    if load_pretrain:           
        if load_pretrained_model(model, pretrained_model):
            model.to(device)

    train(load_pretrain, model, train_dataloader, val_dataloader, criterion, optimizer, device, num_epochs, checkpoint_dir, patience, min_delta)
    
    # Uložení modelu po trénování
    save_final_model(model, out_model)
    logging.info(f"Model size: {os.path.getsize(out_model) / (1024 * 1024):.2f} MB")