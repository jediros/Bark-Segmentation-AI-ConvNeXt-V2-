import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchmetrics.classification import BinaryJaccardIndex
from model import WoodUnetPlusPlusV2
from losses import CombinedLoss

class WoodSegmentationModule(pl.LightningModule):
    def __init__(self, lr=1e-4, num_classes=1):
        super().__init__()
        # Guardamos hiperparámetros (lr, num_classes)
        self.save_hyperparameters()
        
        # Inicializamos el modelo (asegúrate de que model.py use smp.Unet)
        self.model = WoodUnetPlusPlusV2(num_classes=self.hparams.num_classes)
        
        # Pérdida combinada (configurada en modo 'binary')
        self.loss_fn = CombinedLoss()
        
        # Métrica IoU específica para casos binarios
        self.val_iou = BinaryJaccardIndex()

    def forward(self, x): 
        return self.model(x)

    def training_step(self, batch, batch_idx):
        img, mask = batch
        logits = self(img)
        
        # IMPORTANTE: Para pérdida binaria, la máscara debe ser [B, 1, H, W] y tipo float
        mask = mask.unsqueeze(1).float() 
        
        # Validación de seguridad para evitar el RuntimeError de dimensiones
        if logits.shape != mask.shape:
            # Si esto ocurre, el problema está en encoder_depth o decoder_channels en model.py
            raise RuntimeError(f"Discrepancia de formas: Logits {logits.shape} vs Máscara {mask.shape}")
            
        loss = self.loss_fn(logits, mask)
        
        # Logging
        self.log("train_loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        img, mask = batch
        logits = self(img)
        mask_expanded = mask.unsqueeze(1).float()
        
        loss = self.loss_fn(logits, mask_expanded)
        
        # PREDICCIÓN BINARIA:
        # Convertimos logits a probabilidades (0-1) con Sigmoid
        probs = torch.sigmoid(logits).squeeze(1)
        # Umbral 0.5 para decidir si es Corteza (1) o Fondo (0)
        preds = (probs > 0.5).long()
        
        # Actualizamos métrica
        self.val_iou.update(preds, mask)
        
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        self.log("val_iou", self.val_iou, prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        # AdamW es el optimizador recomendado para ConvNeXt V2
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=self.hparams.lr, 
            weight_decay=1e-4
        )
        return optimizer