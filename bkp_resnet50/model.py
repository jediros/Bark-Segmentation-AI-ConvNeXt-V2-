import segmentation_models_pytorch as smp
import torch.nn as nn

class WoodUnetPlusPlusV2(nn.Module):
    def __init__(self, num_classes=1): # <--- Forzamos 1 por defecto
        super().__init__()
        self.model = smp.UnetPlusPlus(
            encoder_name="resnet50", 
            encoder_weights="imagenet",
            in_channels=3,
            classes=num_classes, 
        )

    def forward(self, x):
        return self.model(x)