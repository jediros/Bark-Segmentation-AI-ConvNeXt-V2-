import torch
import cv2
import numpy as np
import os
import argparse
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
from train import WoodSegmentationModule

def predict_wood(ckpt_path, image_path, img_size=640, output_dir="results"):
    # 1. Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(output_dir, exist_ok=True)

    # 2. Load model from checkpoint
    model = WoodSegmentationModule.load_from_checkpoint(ckpt_path, num_classes=1)
    model.to(device).eval()

    # 3. Read and prepare image
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]

    # Validation transform — must match training img_size
    transform = A.Compose([
        A.Resize(img_size, img_size),           # consistent with training
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    
    input_tensor = transform(image=image_rgb)['image'].unsqueeze(0).to(device)

    # 4. Inference
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.sigmoid(logits).squeeze()
        mask = (probs > 0.5).cpu().numpy().astype(np.uint8)

    # 5. Resize mask back to original image dimensions
    mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

    # 6. Create overlay — bark regions highlighted in green
    overlay = image_rgb.copy()
    overlay[mask_resized == 1] = [0, 255, 0]
    combined = cv2.addWeighted(image_rgb, 0.7, overlay, 0.3, 0)

    # 7. Save and display results
    img_name = os.path.basename(image_path)
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(image_rgb); plt.title("Original Image"); plt.axis("off")
    
    plt.subplot(1, 3, 2)
    plt.imshow(mask_resized, cmap="gray"); plt.title("Predicted Mask"); plt.axis("off")
    
    plt.subplot(1, 3, 3)
    plt.imshow(combined); plt.title("Overlay Result"); plt.axis("off")
    
    save_path = os.path.join(output_dir, f"pred_{img_name}")
    plt.savefig(save_path, bbox_inches='tight')
    print(f"✅ Prediction saved to: {save_path}")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bark Segmentation Inference — UNet++ ConvNeXt V2")
    parser.add_argument("--ckpt",       type=str, required=True,    help="Path to .ckpt checkpoint file")
    parser.add_argument("--image",      type=str, required=True,    help="Path to input image")
    parser.add_argument("--img_size",   type=int, default=640,      help="Must match training img_size")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory to save predictions")
    args = parser.parse_args()

    if os.path.exists(args.ckpt):
        predict_wood(args.ckpt, args.image, img_size=args.img_size, output_dir=args.output_dir)
    else:
        print(f"❌ Checkpoint not found: {args.ckpt}")