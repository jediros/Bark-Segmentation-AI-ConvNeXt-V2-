import argparse
import os
import torch # <--- IMPORTANTE
import pytorch_lightning as pl
from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import DataLoader
from data_loader import WoodDataset, get_transforms
from train import WoodSegmentationModule

# LIMITADORES DE HARDWARE PARA EVITAR REINICIOS
torch.set_num_threads(4) # Usa solo 4 hilos (evita el 100% de carga constante)
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

def main(args):
    train_trans, val_trans = get_transforms(img_size=args.img_size)
    
    train_ds = WoodDataset(os.path.join(args.data_path, "images/train"), 
                           os.path.join(args.data_path, "masks/train"), transform=train_trans)
    val_ds = WoodDataset(os.path.join(args.data_path, "images/valid"), 
                         os.path.join(args.data_path, "masks/valid"), transform=val_trans)

    # num_workers=0 es vital en CPU para no saturar la RAM
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, num_workers=0)

    # Cargamos el módulo con ConvNeXt V2 (num_classes=1)
    model = WoodSegmentationModule(lr=args.lr, num_classes=args.num_classes)
    
    logger = MLFlowLogger(experiment_name="Wood_ConvNeXtV2_Bark", tracking_uri="http://0.0.0.0:5015")

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="cpu",
        devices=1,
        logger=logger,
        callbacks=[
            ModelCheckpoint(monitor="val_iou", mode="max", save_top_k=1, filename="best_v2"),
            EarlyStopping(monitor="val_iou", patience=15, mode="max")
        ]
    )

    trainer.fit(model, train_loader, val_loader)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="./data")
    parser.add_argument("--batch_size", type=int, default=1) # Empezamos con 1 por seguridad
    parser.add_argument("--img_size", type=int, default=640)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--num_classes", type=int, default=1) 
    args = parser.parse_args()
    
    main(args)