# src/losses.py
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


class CombinedLoss(nn.Module):
    """
    Pérdida combinada: Dice + Focal + Lovász (modo binario).

    Todas las pérdidas reciben:
      - y_pred: logits con shape (B, 1, H, W)  — SIN sigmoid aplicado
      - y_true: máscara con shape (B, 1, H, W)  — valores 0 o 1, float

    Pesos: Dice 40% | Focal 30% | Lovász 30%
    """

    def __init__(self):
        super().__init__()
        self.dice = smp.losses.DiceLoss(
            mode='binary',
            from_logits=True,
        )
        self.focal = smp.losses.FocalLoss(
            mode='binary',
        )
        self.lovasz = smp.losses.LovaszLoss(
            mode='binary',
            from_logits=True,
            per_image=True,  # Más estable con batch_size=1
        )

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        """
        Args:
            y_pred: Tensor de logits — shape (B, 1, H, W)
            y_true: Tensor de máscara — shape (B, 1, H, W), dtype float

        Returns:
            total_loss: pérdida combinada (escalar)
            components: dict con las pérdidas individuales para loguear en MLflow
        """
        assert y_pred.shape == y_true.shape, (
            f"❌ Shape mismatch en CombinedLoss: "
            f"y_pred={y_pred.shape} vs y_true={y_true.shape}"
        )

        dice_loss   = self.dice(y_pred, y_true)
        focal_loss  = self.focal(y_pred, y_true)
        lovasz_loss = self.lovasz(y_pred, y_true)

        total_loss = 0.4 * dice_loss + 0.3 * focal_loss + 0.3 * lovasz_loss

        components = {
            "dice_loss":   dice_loss,
            "focal_loss":  focal_loss,
            "lovasz_loss": lovasz_loss,
        }

        return total_loss, components