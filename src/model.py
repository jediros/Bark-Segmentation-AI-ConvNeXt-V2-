import segmentation_models_pytorch as smp
import torch.nn as nn

class WoodUnetPlusPlusV2(nn.Module):
    def __init__(self, num_classes=1): 
        super().__init__()
        
        # Usamos smp.Unet en lugar de UnetPlusPlus para total compatibilidad
        # con la estructura de 4 etapas de ConvNeXt V2
        self.model = smp.Unet(
            encoder_name="tu-convnextv2_tiny", 
            encoder_weights="imagenet",
            in_channels=3,
            classes=num_classes,
            encoder_depth=4,                # ConvNeXt V2 tiene 4 etapas
            decoder_channels=(192, 96, 64, 32) # Canales ajustados al backbone
        )

    def forward(self, x):
        return self.model(x)