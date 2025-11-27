# Fast Point-Supervised Segmentation Pipeline – LoveDA Rural Dataset

## Overview
This project implements a fast and optimized training pipeline for **semantic segmentation** using sparse point-level supervision. It is designed for the **LoveDA Rural dataset** and leverages a **Unet++** segmentation model with an **EfficientNet-B3** encoder.

The main goal is to study how the number of labeled points per image affects model performance while keeping the training pipeline fast and lightweight.

---

## Features
- **Partial Point Supervision**: Train using a small number of labeled pixels (points) per image.
- **Optimized Dataset Loading**: Fast loading with simplified augmentations.
- **Partial Focal Loss**: Custom loss function focusing on sparse labeled points.
- **Fast Experiments**: Reduced epochs and batch sizes for quick ablation studies.
- **Automatic Validation & Metrics**: Computes **mIoU** on full masks.
- **Result Visualization**: Generates plots of mIoU versus number of points.

---

## Requirements
- Python 3.9+
- PyTorch 2.x
- torchmetrics
- segmentation-models-pytorch
- albumentations
- opencv-python
- tqdm
- matplotlib
- pandas
- seaborn

### Install Dependencies
```bash
pip install torch torchvision torchaudio
pip install torchmetrics segmentation-models-pytorch albumentations opencv-python tqdm matplotlib pandas seaborn
python3 main.py
```