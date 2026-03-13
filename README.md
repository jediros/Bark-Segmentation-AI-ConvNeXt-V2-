# 🌲 Bark Segmentation AI — ConvNeXt V2 + UNet++

> Semantic segmentation of bark regions in wood trunk images using a custom UNet++ architecture with ConvNeXt V2 backbone. Built as a production-ready ML pipeline with full experiment tracking, modular design, and CPU-optimized training.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange?logo=pytorch)
![Lightning](https://img.shields.io/badge/PyTorch_Lightning-2.x-purple)
![MLflow](https://img.shields.io/badge/MLflow-tracked-green?logo=mlflow)
![Docker](https://img.shields.io/badge/Docker-devcontainer-blue?logo=docker)

---

## 🎯 Problem Statement

Detecting and segmenting bark regions in wood trunk images is a critical task in the **forestry and wood processing industry**. Accurate bark segmentation enables automated quality control, volume estimation, and material classification — tasks traditionally performed manually by skilled operators.

This project tackles the challenge of **training a high-performance segmentation model with a limited dataset** (<50 images), a constraint common in industrial computer vision applications.

---

## 🏗️ Architecture

The model combines two powerful components:

```
Input Image (640×640)
        │
        ▼
┌───────────────────┐
│  ConvNeXt V2      │  ← Backbone preentrenado en ImageNet-22K
│  (Feature Extractor)│    Extrae features semánticas ricas
└────────┬──────────┘
         │  Skip connections
         ▼
┌───────────────────┐
│    UNet++         │  ← Decoder con dense skip connections
│  (Segmentation)   │    Mayor capacidad que UNet clásico
└────────┬──────────┘
         │
         ▼
  Máscara binaria (bark / no-bark)
```

### ¿Por qué UNet++ y no UNet?

UNet++ introduce **dense nested skip connections** entre el encoder y decoder. Esto permite que el modelo aprenda representaciones a múltiples escalas simultáneamente — especialmente relevante para texturas complejas como la corteza de madera, donde la información de bordes finos y regiones amplias debe integrarse.

### ¿Por qué ConvNeXt V2 como backbone?

ConvNeXt V2 (Meta AI, 2023) es un backbone convolucional moderno que iguala o supera a los Vision Transformers en tareas de dense prediction, con menor costo computacional. Al estar preentrenado en ImageNet-22K, aporta representaciones visuales ricas que permiten **hacer transfer learning efectivo con pocos datos**.

---

## ⚙️ Training Pipeline

### Loss Function — CombinedLoss

Se utiliza una función de pérdida combinada que integra:

- **Binary Cross-Entropy (BCE)** — Penaliza pixel a pixel
- **Dice Loss** — Optimiza directamente el solapamiento entre predicción y ground truth

Esta combinación es estándar en segmentación médica e industrial, donde el desbalance de clases es frecuente (más fondo que objeto de interés).

### Optimizer & Scheduler

| Componente | Configuración | Justificación |
|---|---|---|
| Optimizer | AdamW, lr=2e-4 | Mejor generalización que Adam clásico |
| Weight decay | 1e-4 | Regularización para dataset pequeño |
| Scheduler | CosineAnnealingLR | Convergencia suave, evita mínimos locales |
| Early Stopping | patience=15, monitor=val_iou | Detiene el entrenamiento cuando el IoU no mejora |

### CPU Optimization

El pipeline está optimizado para entrenamiento en CPU:

```python
torch.set_num_threads(4)   # Evita saturación de recursos
batch_size = 1             # Minimiza uso de RAM
num_workers = 0            # Estabilidad en CPU
```

---

## 📊 Experiment Tracking

Todos los experimentos se registran automáticamente en **MLflow**, incluyendo:

- Hiperparámetros (lr, batch_size, img_size, epochs)
- Métricas por época (train_loss, val_loss, val_iou, val_dice, lr)
- Artefactos visuales: comparación imagen original / máscara GT / predicción
- Checkpoints del mejor modelo (monitoreado por val_iou)

```bash
# Lanzar MLflow UI
mlflow ui --port 5015
```

### Experimentos realizados

| Run | LR | img_size | Augmentation | val_iou | val_dice |
|---|---|---|---|---|---|
| baseline | 2e-4 | 640 | Standard | — | — |
| low_lr | 5e-5 | 640 | Standard | — | — |
| small_img | 2e-4 | 512 | Standard | — | — |

> Los resultados se actualizan tras completar los runs de entrenamiento.

---

## 🗂️ Project Structure

```
wood_project/
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
│   ├── data_loader.py    # Dataset + augmentation pipeline
│   ├── losses.py         # CombinedLoss (BCE + Dice)
│   ├── model.py          # UNet++ + ConvNeXt V2
│   ├── train.py          # LightningModule (training loop)
│   ├── main.py           # Entry point + MLflow + argparse
│   └── predict.py        # Inference pipeline
├── predictions/          # Visualizaciones por época (auto-generadas)
├── Dockerfile
└── requirements.txt
```

---

## 🚀 Quickstart

### Opción 1 — DevContainer (recomendado)

```bash
# 1. Clonar el repositorio
git clone https://github.com/jediros/Bark-Segmentation-AI-ConvNeXt-V2-.git
cd Bark-Segmentation-AI-ConvNeXt-V2-

# 2. Abrir en VS Code con DevContainer
# → "Reopen in Container" desde VS Code

# 3. Lanzar MLflow
mlflow server --host 0.0.0.0 --port 5015 &

# 4. Entrenar
python src/main.py --data_path ./data --epochs 100 --lr 2e-4
```

### Opción 2 — Docker

```bash
docker build -t bark-seg .
docker run bark-seg python src/main.py --data_path ./data
```

### Argumentos disponibles

| Argumento | Default | Descripción |
|---|---|---|
| `--data_path` | `./data` | Ruta al dataset |
| `--batch_size` | `1` | Batch size (CPU: mantener en 1) |
| `--img_size` | `640` | Resolución de entrada |
| `--lr` | `2e-4` | Learning rate inicial |
| `--epochs` | `100` | Máximo de épocas |
| `--num_classes` | `1` | Clases (1 = segmentación binaria) |

---

## 🔬 Key Technical Decisions

**¿Por qué PyTorch Lightning?**
Elimina el boilerplate del training loop manteniendo control total. Facilita la integración con MLflow, callbacks y reproducibilidad.

**¿Por qué segmentación binaria y no multiclase?**
El problema es inherentemente binario: corteza / no-corteza. Un enfoque binario es más robusto con datasets pequeños que forzar multiclase artificialmente.

**¿Por qué guardar visualizaciones solo cuando mejora val_loss?**
Reduce I/O y mantiene solo las predicciones relevantes. Permite ver visualmente la progresión del modelo a lo largo del entrenamiento de forma eficiente.

**¿Por qué EarlyStopping sobre val_iou y no val_loss?**
IoU (Jaccard Index) mide directamente la calidad de la segmentación. Optimizar sobre la métrica de negocio real, no sobre el proxy de la loss, da mejores modelos en producción.

---

## 🧰 Tech Stack

| Categoría | Tecnología |
|---|---|
| Framework DL | PyTorch + PyTorch Lightning |
| Experiment Tracking | MLflow |
| Métricas | TorchMetrics |
| Containerización | Docker + DevContainers |
| Visualización | Matplotlib |

---

## 👤 Author

**jediros** — MLOps Engineer especializado en Computer Vision  
[GitHub](https://github.com/jediros)

---

> *Este proyecto demuestra cómo construir un pipeline de segmentación semántica production-ready en un dominio industrial con datos limitados, priorizando reproducibilidad, trazabilidad experimental y decisiones técnicas justificadas.*
