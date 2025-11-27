# fixed_fast_pipeline.py - Error Fixed + Optimized for Speed
import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from torchmetrics.classification import MulticlassJaccardIndex
import segmentation_models_pytorch as smp

# ==================== 1. Simplified Partial Focal Loss ====================
class PartialFocalLoss(nn.Module):
    def __init__(self, ignore_index=-1, gamma=2):
        super().__init__()
        self.ignore_index = ignore_index
        self.gamma = gamma

    def forward(self, logits, targets):
        B, C, H, W = logits.shape
        logits_flat = logits.permute(0, 2, 3, 1).reshape(-1, C)
        targets_flat = targets.view(-1)

        mask = targets_flat != self.ignore_index
        if not mask.any():
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        valid_logits = logits_flat[mask]
        valid_targets = targets_flat[mask]

        ce_loss = F.cross_entropy(valid_logits, valid_targets, reduction='none')
        probs = F.softmax(valid_logits, dim=1)
        probs_true = probs[torch.arange(valid_targets.size(0)), valid_targets]
        focal_weight = (1 - probs_true) ** self.gamma

        return (focal_weight * ce_loss).mean()

# ==================== 2. Fast Dataset (No Semi-Supervised) ====================
class FastLoveDADataset(Dataset):
    def __init__(self, split="Train", num_points=10):
        self.img_dir = f"dataset/{split}/Rural/images_png"
        self.mask_dir = f"dataset/{split}/Rural/masks_png"

        # Load valid image-mask pairs
        img_files = {f.split('.')[0] for f in os.listdir(self.img_dir) if f.endswith('.png')}
        mask_files = {f.split('.')[0] for f in os.listdir(self.mask_dir) if f.endswith('.png')}
        self.ids = sorted(list(img_files.intersection(mask_files)))

        print(f"{split}: Loaded {len(self.ids)} images")
        self.num_points = num_points

        # SIMPLIFIED transforms (faster)
        self.transform = A.Compose([
            A.RandomCrop(512, 512),
            A.HorizontalFlip(p=0.3),  # Reduced probability
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        name = self.ids[idx]
        img_path = os.path.join(self.img_dir, name + ".png")
        mask_path = os.path.join(self.mask_dir, name + ".png")

        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path, 0)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Create partial supervision mask
        point_mask = np.full(mask.shape, -1, dtype=np.int16)

        if self.num_points > 0:
            coords = np.where(mask != 255)
            if len(coords[0]) > 0:
                chosen = np.random.choice(len(coords[0]),
                                          min(self.num_points, len(coords[0])),
                                          replace=False)
                ys, xs = coords[0][chosen], coords[1][chosen]
                point_mask[ys, xs] = mask[ys, xs]

        aug = self.transform(image=img, masks=[mask, point_mask])

        # ✅ FIXED: Always return exactly 3 values
        return aug["image"], aug["masks"][1].long(), aug["masks"][0].long()

# ==================== 3. Fast Training Pipeline ====================
def train_fast_pipeline():
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    results = []

    # REDUCED: Only test 2 point settings instead of 4
    for points in [1, 10]: 
        print(f"\n🎯 Training with {points} labeled points per image")

        # Create datasets (NO semi-supervised, NO histogram matching)
        train_ds = FastLoveDADataset("Train", points)
        val_ds = FastLoveDADataset("Val", 0)

        train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=0)

        # Single model (no ensemble)
        model = smp.UnetPlusPlus(
            encoder_name="efficientnet-b3",
            encoder_weights="imagenet",
            classes=7
        ).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        criterion = PartialFocalLoss(ignore_index=-1, gamma=2)
        metric = MulticlassJaccardIndex(num_classes=7, ignore_index=255).to(device)

        best_miou = 0

        for epoch in range(8):
            model.train()
            total_loss = 0

            for img, point_target, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
                img, point_target = img.to(device), point_target.to(device)
                logits = model(img)
                loss = criterion(logits, point_target)
                total_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Validate every 2 epochs (instead of 6)
            if epoch % 2 == 1 or epoch == 7:
                model.eval()
                metric.reset()
                with torch.no_grad():
                    # ✅ FIXED: Proper unpacking that always works
                    for batch in val_loader:
                        # Always get exactly 3 values: img, point_mask, full_mask
                        img, _, full_mask = batch
                        img, full_mask = img.to(device), full_mask.to(device)

                        # NO TTA for speed
                        pred = model(img).argmax(1)
                        metric.update(pred, full_mask)

                miou = metric.compute().item()
                print(f"   → Val mIoU: {miou:.4f}, Train Loss: {total_loss/len(train_loader):.4f}")

                if miou > best_miou:
                    best_miou = miou
                    torch.save(model.state_dict(), f"fast_best_{points}pts.pth")

        results.append({"Points": points, "mIoU": round(best_miou, 4)})

    # ==================== Plot Results ====================
    df = pd.DataFrame(results)
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x="Points", y="mIoU", marker="o", linewidth=3, markersize=10)
    plt.title("Fast Point Supervision Study – LoveDA Rural Dataset", fontsize=16)
    plt.xlabel("Number of Labeled Points per Image")
    plt.ylabel("Validation mIoU")
    plt.grid(True, alpha=0.3)
    plt.savefig("fast_ablation_curve.png", dpi=300, bbox_inches='tight')
    plt.show()

    print("\n✅ FAST PIPELINE FINISHED!")
    print("📊 Results:", results)

if __name__ == "__main__":
    train_fast_pipeline()