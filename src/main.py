import argparse
import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import DataLoader
from data_loader import WoodDataset, get_transforms
from train import WoodSegmentationModule

# CPU LIMITERS — prevents system overload during CPU-only training
torch.set_num_threads(4)
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

def main(args):
    train_trans, val_trans = get_transforms(img_size=args.img_size)
    
    train_ds = WoodDataset(os.path.join(args.data_path, "images/train"), 
                           os.path.join(args.data_path, "masks/train"), transform=train_trans)
    val_ds = WoodDataset(os.path.join(args.data_path, "images/valid"), 
                         os.path.join(args.data_path, "masks/valid"), transform=val_trans)

    # num_workers=0 is required on CPU to avoid RAM saturation
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, num_workers=0)

    model = WoodSegmentationModule(lr=args.lr, num_classes=args.num_classes)
    
    # Dynamic run name — makes each experiment identifiable at a glance in MLflow UI
    run_name = f"convnextv2_lr{args.lr}_img{args.img_size}_bs{args.batch_size}"

    logger = MLFlowLogger(
        experiment_name="Wood_ConvNeXtV2_Bark",
        tracking_uri="http://0.0.0.0:5015",
        run_name=run_name,
        tags={
            "backbone": "convnextv2_tiny",
            "decoder": "unet++",
            "loss": "dice+focal+lovasz",
            "dataset": "bark_wood",
            "author": "jediros"
        }
    )

    # Log all hyperparameters explicitly so they appear in MLflow UI
    logger.log_hyperparams(vars(args))

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
    parser = argparse.ArgumentParser(description="Bark Segmentation Training — UNet++ ConvNeXt V2")
    parser.add_argument("--data_path",    type=str,   default="./data",  help="Path to dataset root")
    parser.add_argument("--batch_size",   type=int,   default=1,         help="Batch size (keep 1 for CPU)")
    parser.add_argument("--img_size",     type=int,   default=640,       help="Input image resolution")
    parser.add_argument("--lr",           type=float, default=2e-4,      help="Initial learning rate")
    parser.add_argument("--epochs",       type=int,   default=100,       help="Maximum training epochs")
    parser.add_argument("--num_classes",  type=int,   default=1,         help="Number of output classes")
    args = parser.parse_args()
    
    main(args)