import pytorch_lightning as pl
import torch
from torchmetrics.classification import BinaryJaccardIndex
from model import WoodUnetPlusPlusV2
from losses import CombinedLoss

class WoodSegmentationModule(pl.LightningModule):
    def __init__(self, lr=1e-4, num_classes=1):
        super().__init__()
        self.save_hyperparameters()
        self.model = WoodUnetPlusPlusV2(num_classes=1) # <--- Forzado a 1
        self.loss_fn = CombinedLoss()
        self.val_iou = BinaryJaccardIndex()

    def forward(self, x): 
        return self.model(x)

    def training_step(self, batch, batch_idx):
        img, mask = batch
        logits = self(img)
        
        # Ajuste de dimensiones para Loss Binaria
        mask = mask.unsqueeze(1).float() 
        
        # Debug en caso de error:
        if logits.shape != mask.shape:
            print(f"\nERROR DE FORMA: Logits {logits.shape} vs Mask {mask.shape}")
            
        loss = self.loss_fn(logits, mask)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        img, mask = batch
        logits = self(img)
        mask_expanded = mask.unsqueeze(1).float()
        
        loss = self.loss_fn(logits, mask_expanded)
        
        # Predicción binaria
        probs = torch.sigmoid(logits).squeeze(1)
        preds = (probs > 0.5).long()
        
        self.val_iou.update(preds, mask)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_iou", self.val_iou, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)