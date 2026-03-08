# src/data_loader.py
import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

class WoodDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        # Filtrar extensiones comunes
        self.images = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])

    def __len__(self): 
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        image = cv2.imread(os.path.join(self.image_dir, img_name))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Asumimos que la máscara es .png y tiene el mismo nombre
        mask_name = os.path.splitext(img_name)[0] + ".png"
        mask = cv2.imread(os.path.join(self.mask_dir, mask_name), cv2.IMREAD_GRAYSCALE)

        if mask is None:
            raise FileNotFoundError(f"No se encontró la máscara: {mask_name} en {self.mask_dir}")

        # --- RE-MAPEO DE ETIQUETAS (BINARIO) ---
        new_mask = np.zeros_like(mask, dtype=np.uint8)
        new_mask[mask == 0] = 1    # 0 en tu archivo -> Clase 1 (Corteza)
        new_mask[mask == 255] = 0  # 255 en tu archivo -> Clase 0 (Nada)
        # ---------------------------------------

        if self.transform:
            augmented = self.transform(image=image, mask=new_mask)
            image, mask = augmented['image'], augmented['mask']

        return image, mask.long()

# --- ESTA ES LA FUNCIÓN QUE TE FALTABA ---
def get_transforms(img_size=640):
    train_transform = A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=30, p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    val_transform = A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    
    return train_transform, val_transform