# 🌲 Bark Segmentation AI — UNet++ ConvNeXt V2

> Binary semantic segmentation of bark regions in wood trunk images. Built as a production-ready ML pipeline featuring a custom triple-loss function, full experiment tracking with MLflow, modular architecture, and CPU-optimized training — all reproducible in one command via DevContainer.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange?logo=pytorch)
![Lightning](https://img.shields.io/badge/PyTorch_Lightning-2.x-purple)
![MLflow](https://img.shields.io/badge/MLflow-tracked-green?logo=mlflow)
![SMP](https://img.shields.io/badge/segmentation--models--pytorch-used-red)
![Docker](https://img.shields.io/badge/DevContainer-ready-blue?logo=docker)

---

## 🎯 Problem Statement

Detecting and segmenting bark regions in wood trunk images is a critical task in the **forestry and wood processing industry**. Accurate bark segmentation enables automated quality control, volume estimation, and raw material classification — tasks traditionally performed manually by skilled operators.

This project addresses the challenge of building a high-performance segmentation model under **real industrial constraints: a small labeled dataset (<50 images)**. Rather than simplifying the problem, this constraint drives deliberate technical decisions around transfer learning, aggressive augmentation, and robust loss design.

---

## 🏗️ Architecture

```
Input Image (512×512×3)
        │
        ▼
┌──────────────────────┐
│   ConvNeXt V2 Tiny   │  Pretrained encoder (ImageNet)
│   (4-stage backbone) │  Extracts rich semantic features
└──────────┬───────────┘
           │  Dense nested skip connections
           ▼
┌──────────────────────┐
│      UNet++          │  Decoder with nested dense blocks
│  (Segmentation head) │  Multi-scale feature fusion
└──────────┬───────────┘
           │
           ▼
    Binary mask (bark / no-bark)
    
Input resolution is configurable via `--img_size`. Must be divisible by 32. Recommended: 512 (fast) or 768 (quality).
```

### Why UNet++ and not standard UNet?

UNet++ introduces **dense nested skip connections** between encoder and decoder stages. This architecture allows the model to learn feature representations at multiple semantic scales simultaneously — particularly useful for bark textures, where both fine edge details and broad region context are equally important for accurate segmentation.

### Why ConvNeXt V2 as the backbone?

ConvNeXt V2 (Meta AI, 2023) is a modern convolutional backbone that matches or outperforms Vision Transformers on dense prediction tasks at lower computational cost. Pretrained on ImageNet, it provides rich visual representations that enable **effective transfer learning with very few labeled samples** — a critical advantage in this industrial setting.

---

## ⚙️ Training Pipeline

### Loss Function — Triple Combined Loss

One of the key technical decisions in this project is the loss function design. Instead of a standard BCE + Dice combination, this pipeline uses a **three-component loss** specifically designed for imbalanced binary segmentation:

| Loss Component | Weight | Purpose |
|---|---|---|
| **Dice Loss** | 0.4 | Directly optimizes mask overlap (IoU proxy) |
| **Focal Loss** | 0.3 | Down-weights easy background pixels, focuses on hard bark regions |
| **Lovász Loss** | 0.3 | Differentiable surrogate for IoU — optimizes the actual evaluation metric |

The Lovász Loss is particularly notable: it provides a differentiable approximation of the IoU score, effectively allowing the model to optimize directly for the evaluation metric rather than a proxy. This combination is common in competitive medical and industrial segmentation pipelines.

### Optimizer & Scheduler

| Component | Configuration | Rationale |
|---|---|---|
| Optimizer | AdamW, lr=2e-4 | Better generalization than Adam via decoupled weight decay |
| Weight decay | 1e-4 | Essential regularization with small datasets |
| Scheduler | CosineAnnealingLR (T_max=50) | Smooth convergence, avoids sharp local minima |
| Early Stopping | patience=15, monitor=val_iou | Stops on the actual segmentation metric, not loss |

### CPU Optimization

The entire pipeline is designed and validated for CPU-only training:

```python
torch.set_num_threads(4)    # Prevents system overload
batch_size = 1              # Minimizes RAM usage
num_workers = 0             # Avoids multiprocessing instability on CPU
```

### Data Augmentation

Training augmentations are applied via **Albumentations** to artificially expand the dataset and improve generalization:

- Horizontal & vertical flips
- Random 90° rotations
- Shift, scale & rotate (shift=0.1, scale=0.1, rotate=30°)
- Random brightness & contrast
- ImageNet normalization

Validation uses resize + normalization only (no augmentation) to ensure unbiased evaluation.

---

## 📊 Experiment Tracking

Every training run is automatically logged to **MLflow**, including:

- All hyperparameters (`lr`, `batch_size`, `img_size`, `epochs`)
- Per-epoch metrics: `train_loss`, `val_loss`, `val_iou`, `val_dice`, `lr`
- Visual artifacts: original image / ground truth mask / predicted mask comparison (logged on every validation improvement)
- Best model checkpoint (monitored by `val_iou`)

```bash
# Start MLflow UI
mlflow server --host 0.0.0.0 --port 5015
# Open http://localhost:5015
```

### Experiments

| Run | LR | img_size | val_iou | val_dice |
|---|---|---|---|---|
| `convnextv2_lr0.0002_img640_bs1` | 2e-4 | 640 | — | — |
| `convnextv2_lr5e-05_img640_bs1` | 5e-5 | 640 | — | — |
| `convnextv2_lr0.0002_img512_bs1` | 2e-4 | 512 | — | — |

> Results will be updated after completing training runs.

---

## 🗂️ Project Structure

```
wood_project/
├── .devcontainer/
│   ├── devcontainer.json     # DevContainer config (portable mount)
│   └── Dockerfile            # CPU-optimized Python 3.10 environment
├── data/
│   ├── images/
│   │   ├── train/
│   │   ├── valid/
│   │   └── test/
│   └── masks/
│       ├── train/
│       ├── valid/
│       └── test/
├── src/
│   ├── data_loader.py        # WoodDataset + Albumentations transforms
│   ├── losses.py             # CombinedLoss (Dice + Focal + Lovász)
│   ├── model.py              # UNet++ + ConvNeXt V2 via SMP
│   ├── train.py              # LightningModule (training loop + metrics)
│   ├── main.py               # Entry point + MLflow + argparse
│   └── predict.py            # Inference pipeline with overlay visualization
├── predictions/              # Per-epoch visual comparisons (auto-generated)
├── results/                  # Inference outputs
└── requirements.txt
```

---

## 🚀 Quickstart

### Option 1 — DevContainer (recommended)

> Requires: VS Code + Docker + Dev Containers extension

```bash
# 1. Clone the repository
git clone https://github.com/jediros/Bark-Segmentation-AI-ConvNeXt-V2-.git
cd Bark-Segmentation-AI-ConvNeXt-V2-

# 2. Update the data mount in .devcontainer/devcontainer.json (see note below)

# 3. Open in VS Code → "Reopen in Container"

# 4. Start MLflow server
mlflow server --host 0.0.0.0 --port 5015 &

# 5. Run training
python src/main.py --data_path /workspaces/Bark_AI/data --epochs 100 --lr 2e-4 --img_size 640
```

### Option 2 — Docker

```bash
docker build -t bark-seg .
docker run bark-seg python src/main.py --data_path ./data
```

### Training Arguments

| Argument | Default | Description |
|---|---|---|
| `--data_path` | `./data` | Path to dataset root |
| `--batch_size` | `1` | Batch size (keep 1 for CPU) |
| `--img_size` | `512` | Input resolution (must match at inference) |
| `--lr` | `2e-4` | Initial learning rate |
| `--epochs` | `100` | Max training epochs (early stopping applies) |
| `--num_classes` | `1` | Output classes (1 = binary segmentation) |

### Inference

```bash
python src/predict.py \
  --ckpt path/to/best_v2.ckpt \
  --image path/to/image.png \
  --img_size 640 \
  --output_dir results/
```

---

## ⚠️ Dataset & Data Mount

This project was trained on a **proprietary industrial dataset** of wood trunk images collected in a real forestry environment and is not publicly available.

To run this project with your own data, update the mount path in `.devcontainer/devcontainer.json`:

```jsonc
"mounts": [
    "source=/your/local/data/path,target=/workspaces/Bark_AI/data,type=bind,consistency=cached"
]
```

Your data folder must follow this structure:

```
data/
├── images/
│   ├── train/
│   ├── valid/
│   └── test/
└── masks/
    ├── train/
    ├── valid/
    └── test/
```

Masks must be binary PNG files (0 = background, 255 = bark) with the same filename as their corresponding image.

---

## 🔬 Key Design Decisions

**Why PyTorch Lightning?**
Eliminates training loop boilerplate while preserving full control. Seamless integration with MLflow callbacks, checkpointing, and early stopping — without sacrificing flexibility.

**Why binary segmentation?**
The problem is inherently binary: bark vs. no-bark. A binary approach is more robust with small datasets than forcing artificial multiclass splits, and maps directly to the downstream industrial use case.

**Why save visualizations only on val_loss improvement?**
Reduces I/O overhead and keeps only meaningful snapshots. The result is a visual history of the model's progression at each quality milestone — useful for debugging and for demonstrating learning dynamics.

**Why monitor EarlyStopping on val_iou instead of val_loss?**
IoU (Jaccard Index) is the real evaluation metric for segmentation quality. Optimizing the stopping criterion on the business metric — not a proxy loss — produces models that are actually better at the task.

**Why Lovász Loss?**
Unlike Dice or BCE, the Lovász-Softmax loss provides a **differentiable surrogate for the IoU score** itself. This means the model is trained to directly optimize what it will be evaluated on, closing the gap between training objective and evaluation metric.

---

## 🧰 Tech Stack

| Category | Technology |
|---|---|
| Deep Learning | PyTorch + PyTorch Lightning |
| Segmentation | segmentation-models-pytorch (SMP) |
| Experiment Tracking | MLflow |
| Augmentation | Albumentations |
| Metrics | TorchMetrics |
| Containerization | Docker + VS Code DevContainers |
| Visualization | Matplotlib + OpenCV |

---

## 👤 Author

**jediros** — MLOps Engineer · Computer Vision  
[GitHub](https://github.com/jediros)

---

> *This project demonstrates how to build a production-ready semantic segmentation pipeline under real industrial constraints — limited labeled data, CPU-only hardware — while maintaining rigorous experiment tracking, reproducibility, and technically justified design decisions throughout.*