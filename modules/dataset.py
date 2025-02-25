import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class ChangeDetectionDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """Inicializuje dataset pro detekci změn.
        
        Args:
            root_dir (str): Kořenový adresář datasetu obsahující složky t1, t2 a masks.
            transform (callable, optional): Transformace aplikované na obrázky.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.t1_dir = os.path.join(root_dir, 't1')
        self.t2_dir = os.path.join(root_dir, 't2')
        self.mask_dir = os.path.join(root_dir, 'mask')
        self.image_filenames = sorted(os.listdir(self.t1_dir))
    
    def __len__(self):
        return len(self.image_filenames)
    
    def __getitem__(self, idx):
        """Načte dvojici obrázků a odpovídající masku."""
        t1_path = os.path.join(self.t1_dir, self.image_filenames[idx])
        t2_path = os.path.join(self.t2_dir, self.image_filenames[idx])
        mask_path = os.path.join(self.mask_dir, self.image_filenames[idx])
        
        t1 = Image.open(t1_path).convert('L')
        t2 = Image.open(t2_path).convert('L')
        mask = Image.open(mask_path).convert('L')
        
        if self.transform:
            t1 = self.transform(t1)
            t2 = self.transform(t2)
            mask = self.transform(mask)
        
        return t1, t2, mask
