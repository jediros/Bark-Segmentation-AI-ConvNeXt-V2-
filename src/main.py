# src/main.py
import argparse
import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import DataLoader
from data_loader import WoodDataset, get_transforms
from train import WoodSegmentationModule

# Limitadores de hardware para evitar reinicios en CPU
torch.set_num_threads(4)
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"


def main(args):
    train_trans, val_trans = get_transforms(img_size=args.img_size)

    train_ds = WoodDataset(
        os.path.join(args.data_path, "images/train"),
        os.path.join(args.data_path, "masks/train"),
        transform=train_trans
    )
    val_ds = WoodDataset(
        os.path.join(args.data_path, "images/valid"),
        os.path.join(args.data_path, "masks/valid"),
        transform=val_trans
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Cargamos el módulo — img_size queda grabado en el .ckpt via save_hyperparameters()
    model = WoodSegmentationModule(
        lr=args.lr,
        num_classes=args.num_classes,
        img_size=args.img_size,
    )

    # --- NOMBRE DEL RUN: descriptivo para comparar en MLflow UI ---
    # Ejemplo: "convnextv2_bs1_lr2e-04_img512_ep100"
    run_name = (
        f"convnextv2"
        f"_bs{args.batch_size}"
        f"_lr{args.lr:.0e}"
        f"_img{args.img_size}"
        f"_ep{args.epochs}"
    )

    logger = MLFlowLogger(
        experiment_name="Wood_ConvNeXtV2_Bark",
        run_name=run_name,
        tracking_uri="http://0.0.0.0:5015",
        log_model=False,  # El checkpoint ya lo maneja ModelCheckpoint
    )

    # --- TAGS: aparecen como columnas filtrables en MLflow UI ---
    # Se loguean justo después de que el logger crea el run
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="cpu",
        devices=1,
        logger=logger,
        callbacks=[
            ModelCheckpoint(
                monitor="val_iou",
                mode="max",
                save_top_k=1,
                filename="best_v2"
            ),
            EarlyStopping(
                monitor="val_iou",
                patience=15,
                mode="max"
            ),
        ]
    )

    # Loguear tags DESPUÉS de trainer.fit() para que el run_id ya exista
    # Los tags permiten filtrar y comparar runs en la UI de MLflow
    def log_tags():
        logger.experiment.set_tag(logger.run_id, "encoder",         "tu-convnextv2_tiny")
        logger.experiment.set_tag(logger.run_id, "architecture",    "Unet")
        logger.experiment.set_tag(logger.run_id, "img_size",        str(args.img_size))
        logger.experiment.set_tag(logger.run_id, "batch_size",      str(args.batch_size))
        logger.experiment.set_tag(logger.run_id, "loss",            "Dice40+Focal30+Lovasz30")
        logger.experiment.set_tag(logger.run_id, "optimizer",       "AdamW")
        logger.experiment.set_tag(logger.run_id, "scheduler",       "CosineAnnealingLR")
        logger.experiment.set_tag(logger.run_id, "dataset",         os.path.basename(args.data_path))
        logger.experiment.set_tag(logger.run_id, "num_train",       str(len(train_ds)))
        logger.experiment.set_tag(logger.run_id, "num_val",         str(len(val_ds)))

    trainer.fit(model, train_loader, val_loader)
    log_tags()

    print(f"\n✅ Entrenamiento finalizado.")
    print(f"   Run name : {run_name}")
    print(f"   Run ID   : {logger.run_id}")
    print(f"   MLflow UI: http://0.0.0.0:5015")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrenamiento de segmentación de corteza")
    parser.add_argument("--data_path",   type=str,   default="./data")
    parser.add_argument("--batch_size",  type=int,   default=1)
    parser.add_argument("--img_size",    type=int,   default=512)
    parser.add_argument("--lr",          type=float, default=2e-4)
    parser.add_argument("--epochs",      type=int,   default=100)
    parser.add_argument("--num_classes", type=int,   default=1)
    args = parser.parse_args()

    main(args)