import torch
import cv2
import numpy as np
import argparse
from train import WoodSegmentationModule
from data_loader import get_transforms
import matplotlib.pyplot as plt

def predict(ckpt, img_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = WoodSegmentationModule.load_from_checkpoint(ckpt).to(device).eval()
    
    _, val_trans = get_transforms(img_size=640)
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    h, w = img.shape[:2]
    input_tensor = val_trans(image=img_rgb)['image'].unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(input_tensor)
        preds = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()

    mask = cv2.resize(preds.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    
    # Mostrar resultados
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1); plt.imshow(img_rgb); plt.title("Original")
    plt.subplot(1,2,2); plt.imshow(mask, cmap='jet'); plt.title("Predicción")
    plt.show()

if __name__ == "__main__":
    # predict("path/to/best_wood_v2.ckpt", "test_wood.jpg")
    pass
