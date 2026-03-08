import argparse
import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import DataLoader
from data_loader import WoodDataset, get_transforms
from train import WoodSegmentationModule

# LIMITADORES DE HARDWARE PARA EVITAR REINICIOS POR CALOR
torch.set_num_threads(4) 
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

def main(args):
    # 1. Rutas de guardado (Estilo Segformer)
    # Creamos una carpeta 'visual_predictions' dentro de la carpeta del proyecto
    output_dir = os.path.join(os.getcwd(), "visual_predictions")
    os.makedirs(output_dir, exist_ok=True)

    # 2. Configurar transformaciones
    train_trans, val_trans = get_transforms(img_size=args.img_size)
    
    # 3. Preparar Datasets
    train_ds = WoodDataset(os.path.join(args.data_path, "images/train"), 
                           os.path.join(args.data_path, "masks/train"), transform=train_trans)
    val_ds = WoodDataset(os.path.join(args.data_path, "images/valid"), 
                         os.path.join(args.data_path, "masks/valid"), transform=val_trans)

    # 4. DataLoaders (num_workers=0 para estabilidad en CPU)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, num_workers=0)

    # 5. Cargar el módulo con la nueva ruta de salida para imágenes
    # Pasamos lr, num_classes y el nuevo output_dir para las visualizaciones
    model = WoodSegmentationModule(
        lr=args.lr, 
        num_classes=args.num_classes,
        output_dir=output_dir
    )
    
    # 6. Logger (MLFlow)
    logger = MLFlowLogger(experiment_name="Wood_ConvNeXtV2_Bark", tracking_uri="http://0.0.0.0:5015")

    # 7. Trainer
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

    # 8. Iniciar entrenamiento
    print(f"--- Entrenamiento iniciado. Las muestras visuales se guardarán en: {output_dir} ---")
    trainer.fit(model, train_loader, val_loader)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="./data")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--img_size", type=int, default=640)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--num_classes", type=int, default=1) 
    args = parser.parse_args()
    
    main(args)