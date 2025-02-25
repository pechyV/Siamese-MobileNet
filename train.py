import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from modules.dataset import ChangeDetectionDataset
from model.siamese_unet import SiameseUNet, get_model
from modules.utils import visualize_results
from modules.utils import setup_logging
import logging

# Nastavení logování
setup_logging()

def train(model, train_dataloader, val_dataloader, criterion, optimizer, device, num_epochs):
    for epoch in range(num_epochs):
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
        with torch.no_grad():  # Deaktivujeme výpočty gradientů pro validaci
            for t1, t2, mask in val_dataloader:
                t1, t2, mask = t1.to(device), t2.to(device), mask.to(device)
                outputs = model(t1, t2)
                loss = criterion(outputs, mask)
                val_loss += loss.item()

        logging.info(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss/len(val_dataloader):.4f}")

        # Vizualizace výsledků po každé epoše
        with torch.no_grad():  # Deaktivujeme výpočty gradientů pro vizualizaci
            t1, t2, mask = t1[0], t2[0], mask[0]  # První obrázek v dávce
            visualize_results(t1, t2, mask, (outputs > 0.5).float()[0], epoch+1)  # Uložení obrázku pro aktuální epochu !!!včetně prahování (outputs > 0.5).float()

if __name__ == "__main__":

    """ Parametry """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(device)
    learning_rate = 0.001
    num_epochs = 5
    batch_size = 8
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), learning_rate)
    train_root_dir = "./test_dataset/train/"
    val_root_dir = "./test_dataset/val/"
    out_model = "./trained_model/siamese_unet.pth"
    
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    # Načítání trénovacího a validačního datasetu
    train_dataset = ChangeDetectionDataset(train_root_dir, transform=transform)
    val_dataset = ChangeDetectionDataset(val_root_dir, transform=transform)

    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size, shuffle=False)

    # Trénování modelu
    train(model, train_dataloader, val_dataloader, criterion, optimizer, device, num_epochs)
    
    # Uložení modelu po trénování
    torch.save(model.state_dict(), out_model)
    logging.info("Model uložen jako siamese_unet.pth")
