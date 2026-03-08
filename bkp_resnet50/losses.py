import torch.nn as nn
import segmentation_models_pytorch as smp

class CombinedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
        # CRÍTICO: mode='binary' en todas las pérdidas
        self.dice = smp.losses.DiceLoss(mode='binary')
        self.focal = smp.losses.FocalLoss(mode='binary')
        self.lovasz = smp.losses.LovaszLoss(mode='binary')

    def forward(self, y_pred, y_true):
        # y_pred: logits del modelo [B, 1, H, W]
        # y_true: máscara real con ceros y unos [B, 1, H, W]
        
        dice_loss = self.dice(y_pred, y_true)
        focal_loss = self.focal(y_pred, y_true)
        lovasz_loss = self.lovasz(y_pred, y_true)
        
        # Combinación: 40% Dice, 30% Focal, 30% Lovasz
        return 0.4 * dice_loss + 0.3 * focal_loss + 0.3 * lovasz_loss