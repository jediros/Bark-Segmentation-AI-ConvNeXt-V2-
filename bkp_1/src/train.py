import pytorch_lightning as pl
import torch
import os
import matplotlib.pyplot as plt
from torchmetrics.classification import BinaryJaccardIndex, BinaryF1Score
from model import WoodUnetPlusPlusV2
from losses import CombinedLoss

class WoodSegmentationModule(pl.LightningModule):
    def __init__(self, lr=1e-4, num_classes=1, output_dir="predictions"):
        super().__init__()
        self.save_hyperparameters()
        
        # Model & Loss
        self.model = WoodUnetPlusPlusV2(num_classes=num_classes)
        self.loss_fn = CombinedLoss()
        
        # Metrics
        self.val_iou = BinaryJaccardIndex()
        self.val_dice = BinaryF1Score()          # Dice = F1 in binary segmentation

        # Prediction saving logic
        self.output_dir = output_dir
        self.best_val_loss = float("inf")
        self.val_outputs = []

    def forward(self, x): 
        return self.model(x)

    def training_step(self, batch, batch_idx):
        img, mask = batch
        logits = self(img)
        loss = self.loss_fn(logits, mask.unsqueeze(1).float())
        self.log("train_loss", loss, prog_bar=True, on_epoch=True)
        # Log learning rate to track CosineAnnealingLR decay
        self.log("lr", self.optimizers().param_groups[0]["lr"], on_epoch=True, prog_bar=False)
        return loss

    def validation_step(self, batch, batch_idx):
        img, mask = batch
        logits = self(img)
        
        loss = self.loss_fn(logits, mask.unsqueeze(1).float())
        
        # Binary prediction
        probs = torch.sigmoid(logits).squeeze(1)
        preds = (probs > 0.5).long()
        
        # Update metrics
        self.val_iou.update(preds, mask)
        self.val_dice.update(preds, mask)

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_iou", self.val_iou, prog_bar=True)
        self.log("val_dice", self.val_dice, prog_bar=True)   # NEW: Dice score
        
        # Save one sample per epoch for visualization
        if batch_idx == 0:
            self.val_outputs.append({
                "images": img.detach().cpu(), 
                "masks": mask.detach().cpu(), 
                "probs": probs.detach().cpu()
            })
            
        return loss

    def on_validation_epoch_end(self):
        avg_loss = self.trainer.callback_metrics.get("val_loss")
        
        if avg_loss is not None and avg_loss < self.best_val_loss:
            self.best_val_loss = avg_loss
            print(f"\n--- Improvement detected: {self.best_val_loss:.4f}. Saving sample predictions... ---")
            self.save_predictions()
            
        self.val_outputs = []

    def save_predictions(self):
        os.makedirs(self.output_dir, exist_ok=True)
        
        if not self.val_outputs:
            return
            
        batch = self.val_outputs[0]
        images = batch["images"]
        masks = batch["masks"]
        probs = batch["probs"]
        
        num_images = min(images.shape[0], 4)
        
        for i in range(num_images):
            # De-normalize image for visualization
            img_vis = images[i].permute(1, 2, 0).numpy()
            img_vis = (img_vis * [0.229, 0.224, 0.225]) + [0.485, 0.456, 0.406]
            img_vis = img_vis.clip(0, 1)

            pred_mask = (probs[i] > 0.5).numpy().astype(int)

            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            axes[0].imshow(img_vis)
            axes[0].set_title("Original Image")
            axes[1].imshow(masks[i], cmap="gray")
            axes[1].set_title("Ground Truth Mask")
            axes[2].imshow(pred_mask, cmap="gray")
            axes[2].set_title("Prediction (ConvNeXtV2)")
            
            for ax in axes: ax.axis("off")
            
            img_path = os.path.join(self.output_dir, f"epoch_{self.current_epoch}_sample_{i}.png")
            plt.savefig(img_path, bbox_inches="tight")
            plt.close(fig)

            # Log prediction images as MLflow artifacts
            try:
                if self.logger and hasattr(self.logger, 'run_id'):
                    self.logger.experiment.log_artifact(self.logger.run_id, img_path)
            except Exception:
                pass  # Silently skip if MLflow artifact logging is unavailable

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}