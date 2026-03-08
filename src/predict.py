# src/predict.py
import torch
import cv2
import numpy as np
import os
import argparse
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
from train import WoodSegmentationModule

# --- TAMAÑO DE IMAGEN: debe coincidir exactamente con el usado en entrenamiento ---
# Cambia este valor si entrenaste con un img_size diferente (ej: 512, 768, etc.)
DEFAULT_IMG_SIZE = 512


def predict_wood(ckpt_path, image_path, output_dir="results", img_size=DEFAULT_IMG_SIZE):
    # 1. Validaciones de entrada
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"❌ No se encontró el checkpoint: '{ckpt_path}'\n"
            f"   Verifica la ruta del archivo .ckpt generado por el entrenamiento."
        )
    if not os.path.exists(image_path):
        raise FileNotFoundError(
            f"❌ No se encontró la imagen: '{image_path}'"
        )

    os.makedirs(output_dir, exist_ok=True)

    # 2. Configuración del dispositivo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Usando dispositivo: {device}")
    print(f"📐 Tamaño de imagen para inferencia: {img_size}x{img_size}")

    # 3. Cargar el modelo desde el checkpoint
    model = WoodSegmentationModule.load_from_checkpoint(ckpt_path, num_classes=1)
    img_size= model.hparams.img_size
    
    model.to(device).eval()
    print("✅ Modelo cargado correctamente.")

    # 4. Leer y validar la imagen
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(
            f"❌ No se pudo cargar la imagen '{image_path}'. El archivo puede estar corrupto."
        )
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]

    # 5. Transformación: usa el MISMO img_size que en entrenamiento
    transform = A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    input_tensor = transform(image=image_rgb)['image'].unsqueeze(0).to(device)

    # 6. Predicción
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.sigmoid(logits).squeeze()
        mask = (probs > 0.5).cpu().numpy().astype(np.uint8)

    # 7. Redimensionar máscara al tamaño original de la foto
    mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

    # 8. Crear imagen combinada (Overlay)
    overlay = image_rgb.copy()
    overlay[mask_resized == 1] = [0, 255, 0]  # Corteza en VERDE brillante
    combined = cv2.addWeighted(image_rgb, 0.7, overlay, 0.3, 0)

    # 9. Guardar y mostrar resultado
    img_name = os.path.basename(image_path)
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(image_rgb)
    plt.title("Imagen Original")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(mask_resized, cmap="gray")
    plt.title("Máscara Predicha")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(combined)
    plt.title("Resultado (Overlay)")
    plt.axis("off")

    save_path = os.path.join(output_dir, f"pred_{img_name}")
    plt.savefig(save_path, bbox_inches='tight')
    print(f"✅ Predicción guardada en: {save_path}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inferencia del modelo de segmentación de corteza")
    parser.add_argument("--ckpt",       type=str, required=True,              help="Ruta al archivo .ckpt del mejor modelo")
    parser.add_argument("--image",      type=str, required=True,              help="Ruta a la imagen a segmentar")
    parser.add_argument("--output_dir", type=str, default="results",          help="Carpeta donde guardar el resultado")
    parser.add_argument("--img_size",   type=int, default=DEFAULT_IMG_SIZE,   help="Tamaño de imagen (debe coincidir con entrenamiento)")
    args = parser.parse_args()

    predict_wood(
        ckpt_path=args.ckpt,
        image_path=args.image,
        output_dir=args.output_dir,
        img_size=args.img_size,
    )