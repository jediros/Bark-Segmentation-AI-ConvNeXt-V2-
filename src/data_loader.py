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
        all_files = [
            f for f in os.listdir(image_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
        ]

        # --- VALIDACIÓN TEMPRANA: verificar que cada imagen tiene su máscara ---
        valid_files = []
        missing_masks = []
        for f in sorted(all_files):
            mask_name = os.path.splitext(f)[0] + ".png"
            mask_path = os.path.join(mask_dir, mask_name)
            if os.path.exists(mask_path):
                valid_files.append(f)
            else:
                missing_masks.append(mask_name)

        if missing_masks:
            print(f"⚠️  Advertencia: {len(missing_masks)} imágenes sin máscara fueron ignoradas:")
            for m in missing_masks[:10]:  # Mostramos máximo 10 para no saturar consola
                print(f"   - {m}")
            if len(missing_masks) > 10:
                print(f"   ... y {len(missing_masks) - 10} más.")

        self.images = valid_files
        print(f"✅ Dataset listo: {len(self.images)} pares imagen/máscara encontrados en '{image_dir}'")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)

        # --- VALIDACIÓN: verificar que la imagen se cargó correctamente ---
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(
                f"No se pudo cargar la imagen: '{img_path}'. "
                f"Verifica que el archivo no esté corrupto."
            )
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Cargar máscara (ya validada en __init__, pero chequeamos igualmente)
        mask_name = os.path.splitext(img_name)[0] + ".png"
        mask_path = os.path.join(self.mask_dir, mask_name)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(
                f"No se pudo cargar la máscara: '{mask_path}'. "
                f"El archivo puede estar corrupto."
            )

        # --- RE-MAPEO DE ETIQUETAS (BINARIO) ---
        # Píxeles con valor 255 en la máscara -> Clase 1 (Corteza)
        # Píxeles con valor   0 en la máscara -> Clase 0 (Fondo/Nada)
        new_mask = np.zeros_like(mask, dtype=np.uint8)
        new_mask[mask == 255] = 1
        new_mask[mask == 0] = 0
        # ----------------------------------------

        if self.transform:
            augmented = self.transform(image=image, mask=new_mask)
            image, mask = augmented['image'], augmented['mask']

        return image, mask.long()


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