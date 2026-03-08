import argparse
import os
import pytorch_lightning as pl
from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import DataLoader
from data_loader import WoodDataset, get_transforms
from train import WoodSegmentationModule

import torch
# Limita a 2 núcleos físicos. Tu VivoBook te lo agradecerá.
torch.set_num_threads(4) 
# Evita que el sistema use toda la memoria de intercambio (Swap)
torch.set_num_interop_threads(4)


def main(args):
    # 1. Configuramos las transformaciones
    train_trans, val_trans = get_transforms(img_size=args.img_size)
    
    # 2. Cargamos los Datasets
    train_ds = WoodDataset(os.path.join(args.data_path, "images/train"), 
                           os.path.join(args.data_path, "masks/train"), transform=train_trans)
    val_ds = WoodDataset(os.path.join(args.data_path, "images/valid"), 
                         os.path.join(args.data_path, "masks/valid"), transform=val_trans)

    # 3. Configuramos los DataLoaders
    # Nota: num_workers=4 es más seguro para evitar que la CPU se trabe en el primer test
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, num_workers=4, pin_memory=False)

    # 4. CREAMOS EL MODELO (Usando el valor que viene de los argumentos)
    print(f"--- Iniciando modelo para {args.num_classes} clase(s) ---")
    model = WoodSegmentationModule(lr=args.lr, num_classes=args.num_classes)
    
    # 5. Configuramos el Logger de MLFlow
    logger = MLFlowLogger(experiment_name="Wood_Bark_Segmentation", tracking_uri="http://0.0.0.0:5015")

    # 6. Configuramos el Trainer
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="cpu", # Forzamos CPU
        devices=1,
        logger=logger,
        callbacks=[
            ModelCheckpoint(monitor="val_iou", mode="max", save_top_k=1, filename="best_wood_v2"),
            EarlyStopping(monitor="val_iou", patience=15, mode="max")
        ]
    )

    # 7. ¡A ENTRENAR!
    trainer.fit(model, train_loader, val_loader)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="./data")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--img_size", type=int, default=640)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--num_classes", type=int, default=1) # Valor por defecto: 1
    
    args = parser.parse_args()
    
    # Llamamos a la función main pasando los argumentos
    main(args)