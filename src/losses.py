import torch.nn as nn
import segmentation_models_pytorch as smp

class CombinedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.dice = smp.losses.DiceLoss(mode='binary')
        self.focal = smp.losses.FocalLoss(mode='binary')
        self.lovasz = smp.losses.LovaszLoss(mode='binary')

    def forward(self, y_pred, y_true):
        # Aseguramos que y_true tenga la misma forma que y_pred
        return 0.4 * self.dice(y_pred, y_true) + 0.3 * self.focal(y_pred, y_true) + 0.3 * self.lovasz(y_pred, y_true)