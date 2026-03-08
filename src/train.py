# src/train.py
import pytorch_lightning as pl
import torch
import torch.nn as nn
import os
import matplotlib.pyplot as plt
from torchmetrics.classification import BinaryJaccardIndex
from model import WoodUnetPlusPlusV2
from losses import CombinedLoss


class WoodSegmentationModule(pl.LightningModule):
    def __init__(self, lr=1e-4, num_classes=1, img_size=512, output_dir="predictions"):
        super().__init__()
        # Guarda lr, num_classes, img_size → quedan grabados en el .ckpt
        # Así predict.py puede leer img_size directamente del checkpoint
        self.save_hyperparameters()

        # Modelo y Pérdida
        self.model = WoodUnetPlusPlusV2(num_classes=num_classes)
        self.loss_fn = CombinedLoss()

        # Métricas
        self.val_iou = BinaryJaccardIndex()

        # Lógica de guardado de imágenes
        self.output_dir = output_dir
        self.best_val_loss = float("inf")
        self.val_outputs = []

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        img, mask = batch
        logits = self(img)

        # CombinedLoss ahora retorna (total_loss, components)
        loss, components = self.loss_fn(logits, mask.unsqueeze(1).float())

        # --- Métricas de entrenamiento para MLflow ---
        self.log("train_loss",        loss,                      prog_bar=True,  on_epoch=True, on_step=False)
        self.log("train_dice_loss",   components["dice_loss"],   prog_bar=False, on_epoch=True, on_step=False)
        self.log("train_focal_loss",  components["focal_loss"],  prog_bar=False, on_epoch=True, on_step=False)
        self.log("train_lovasz_loss", components["lovasz_loss"], prog_bar=False, on_epoch=True, on_step=False)

        return loss

    def validation_step(self, batch, batch_idx):
        img, mask = batch
        logits = self(img)

        # CombinedLoss ahora retorna (total_loss, components)
        loss, components = self.loss_fn(logits, mask.unsqueeze(1).float())

        # Predicción binaria
        probs = torch.sigmoid(logits).squeeze(1)
        preds = (probs > 0.5).long()

        self.val_iou.update(preds, mask)

        # --- Métricas de validación para MLflow ---
        self.log("val_loss",        loss,                      prog_bar=True)
        self.log("val_iou",         self.val_iou,              prog_bar=True)
        self.log("val_dice_loss",   components["dice_loss"],   prog_bar=False)
        self.log("val_focal_loss",  components["focal_loss"],  prog_bar=False)
        self.log("val_lovasz_loss", components["lovasz_loss"], prog_bar=False)

        # Guardamos muestra del primer batch para visualizar al final de época
        if batch_idx == 0:
            self.val_outputs.append({
                "images": img.detach().cpu(),
                "masks":  mask.detach().cpu(),
                "probs":  probs.detach().cpu()
            })

        return loss

    def on_validation_epoch_end(self):
        # --- Loguear el learning rate actual (útil para ver el scheduler) ---
        current_lr = self.optimizers().param_groups[0]['lr']
        self.log("learning_rate", current_lr, prog_bar=False)

        # Guardar imágenes si hubo mejora
        avg_loss = self.trainer.callback_metrics.get("val_loss")
        if avg_loss is not None and avg_loss < self.best_val_loss:
            self.best_val_loss = avg_loss
            print(f"\n--- Mejora detectada: {self.best_val_loss:.4f}. Guardando imágenes de muestra... ---")
            self.save_predictions()

        self.val_outputs = []

    def save_predictions(self):
        os.makedirs(self.output_dir, exist_ok=True)

        if not self.val_outputs:
            return

        batch = self.val_outputs[0]
        images = batch["images"]
        masks  = batch["masks"]
        probs  = batch["probs"]

        num_images = min(images.shape[0], 4)

        for i in range(num_images):
            # Des-normalizar imagen para visualización
            img_vis = images[i].permute(1, 2, 0).numpy()
            img_vis = (img_vis * [0.229, 0.224, 0.225]) + [0.485, 0.456, 0.406]
            img_vis = img_vis.clip(0, 1)

            pred_mask = (probs[i] > 0.5).numpy().astype(int)

            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            axes[0].imshow(img_vis)
            axes[0].set_title("Imagen Original")
            axes[1].imshow(masks[i], cmap="gray")
            axes[1].set_title("Máscara Real (GT)")
            axes[2].imshow(pred_mask, cmap="gray")
            axes[2].set_title("Predicción (ConvNeXtV2)")

            for ax in axes:
                ax.axis("off")

            plt.savefig(
                os.path.join(self.output_dir, f"epoch_{self.current_epoch}_sample_{i}.png")
            )
            plt.close(fig)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=1e-4
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}