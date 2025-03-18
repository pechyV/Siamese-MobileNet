import torch
import os
import logging
import matplotlib.pyplot as plt

def save_checkpoint(model, optimizer, epoch, checkpoint_dir="checkpoints"):
    """Ukládá model a optimizer do souboru pro pozdější pokračování v tréninku."""
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pth")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, checkpoint_path)
    logging.info(f"Checkpoint saved: {checkpoint_path}")

def load_checkpoint(model, optimizer, checkpoint_path):
    """Načte model a optimizer ze souboru."""
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    logging.info(f"Checkpoint loaded {checkpoint_path}, epoch {checkpoint['epoch']}")
    return checkpoint['epoch']

def save_final_model(model, output_path="./trained_model/siamese_unet.pth"):
    """Ukládá finální model po dokončení trénování."""
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    torch.save(model.state_dict(), output_path)
    logging.info(f"Final model saved: {output_path}")

def load_pretrained_model(model, model_path):
    """Načte předtrénovaný model ze souboru."""
    if not os.path.exists(model_path):
        logging.error(f"Soubor {model_path} neexistuje!")
        return False
    model.load_state_dict(torch.load(model_path))
    logging.info(f"Pretrained model loaded: {model_path}")
    return True

def setup_logging(log_dir="./logs/"):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logging.basicConfig(
        level=logging.INFO,  
        format="%(asctime)s - %(message)s", 
        handlers=[logging.FileHandler(log_dir + "log.txt"), logging.StreamHandler()]
    )

def visualize_results(t1, t2, mask, prediction, epoch, idx, save_dir="./visualizations"):
    """Vizualizuje vstupní snímky, ground truth masku a predikci, a ukládá je jako obrázky."""
    
    epoch_dir = os.path.join(save_dir, f"epoch_{epoch}")
    if not os.path.exists(epoch_dir):
        os.makedirs(epoch_dir)

    if isinstance(t1, torch.Tensor):
        t1 = t1.cpu().detach().numpy()
    if isinstance(t2, torch.Tensor):
        t2 = t2.cpu().detach().numpy()
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().detach().numpy()
    if isinstance(prediction, torch.Tensor):
        prediction = prediction.cpu().detach().numpy()

    # Vytvoření subgrafu pro zobrazení
    fig, axs = plt.subplots(1, 4, figsize=(12, 4))
    axs[0].imshow(t1.squeeze(), cmap='gray')
    axs[0].set_title("T1 Image")
    axs[1].imshow(t2.squeeze(), cmap='gray')
    axs[1].set_title("T2 Image")
    axs[2].imshow(mask.squeeze(), cmap='gray')
    axs[2].set_title("Ground Truth Mask")
    axs[3].imshow(prediction.squeeze(), cmap='gray')
    axs[3].set_title("Prediction")
    
    for ax in axs:
        ax.axis("off")
    
    # Uložení obrázku jako soubor
    file_path = os.path.join(epoch_dir, f"sample_{idx}.png")
    plt.savefig(file_path)
    plt.close(fig)


