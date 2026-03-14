import torch
import cv2
import numpy as np
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
from train import WoodSegmentationModule

def predict_wood(ckpt_path, image_path, output_dir="results"):
    # 1. Configuración del dispositivo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(output_dir, exist_ok=True)

    # 2. Cargar el modelo desde el checkpoint
    # Asegúrate de que num_classes=1 coincida con tu entrenamiento
    model = WoodSegmentationModule.load_from_checkpoint(ckpt_path, num_classes=1)
    model.to(device).eval()

    # 3. Leer y preparar la imagen
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]

    # Transformación idéntica a la de validación
    transform = A.Compose([
        A.Resize(512, 512),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    
    input_tensor = transform(image=image_rgb)['image'].unsqueeze(0).to(device)

    # 4. Predicción
    with torch.no_grad():
        logits = model(input_tensor)
        # Como es binario, usamos Sigmoide + Umbral 0.5
        probs = torch.sigmoid(logits).squeeze()
        mask = (probs > 0.5).cpu().numpy().astype(np.uint8)

    # 5. Redimensionar máscara al tamaño original de la foto
    mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

    # 6. Crear una imagen combinada (Overlay) para ver el resultado
    overlay = image_rgb.copy()
    overlay[mask_resized == 1] = [0, 255, 0] # Pintamos la corteza de VERDE brillante
    
    combined = cv2.addWeighted(image_rgb, 0.7, overlay, 0.3, 0)

    # 7. Guardar y Mostrar
    img_name = os.path.basename(image_path)
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(image_rgb); plt.title("Imagen Original"); plt.axis("off")
    
    plt.subplot(1, 3, 2)
    plt.imshow(mask_resized, cmap="gray"); plt.title("Máscara Predicha"); plt.axis("off")
    
    plt.subplot(1, 3, 3)
    plt.imshow(combined); plt.title("Resultado (Overlay)"); plt.axis("off")
    
    save_path = os.path.join(output_dir, f"pred_{img_name}")
    plt.savefig(save_path, bbox_inches='tight')
    print(f"✅ Predicción guardada en: {save_path}")
    plt.show()

if __name__ == "__main__":
    # Cambia esto por la ruta real de tu archivo .ckpt que generó el entrenamiento
    # Suele estar en lightning_logs/version_X/checkpoints/best_model.ckpt
    CKPT = "/workspaces/Bark_AI/3/34163f26205a4612a79bc2dd7447c725/checkpoints/best_v2.ckpt"
    IMG = "/workspaces/Bark_AI/data/images/test/bark_export_5B_800N.png" 
    
    if os.path.exists(CKPT):
        predict_wood(CKPT, IMG)
    else:
        print("❌ No se encontró el archivo .ckpt. Revisa la ruta.")