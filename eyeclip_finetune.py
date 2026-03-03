import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, recall_score, f1_score

import torchvision.transforms as T

# EyeCLIP repo import
import sys
import eyeclip.clip as clip


# --------------------------
# Dataset
# --------------------------
class ImageDataset(Dataset):
    def __init__(self, image_paths, labels, augment=False):
        self.image_paths = image_paths
        self.labels = labels
        self.augment = augment

        # CLIP normalization only (no resize, no crop)
        clip_mean = (0.48145466, 0.4578275, 0.40821073)
        clip_std  = (0.26862953, 0.26130258, 0.27577711)

        self.base_transform = T.Compose([
            T.ToTensor(),
            T.Normalize(clip_mean, clip_std),
        ])

        if augment:
            self.aug = T.Compose([
                T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=(0, 0.1)),
                T.RandomApply([T.GaussianBlur(kernel_size=3)], p=0.5),
                T.RandomRotation(15),
            ])
        else:
            self.aug = None

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        if self.aug:
            image = self.aug(image)
        image_tensor = self.base_transform(image)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return image_tensor, label


# --------------------------
# Model wrapper (EyeCLIP encoder + classifier head)
# --------------------------
class EyeCLIPClassifier(nn.Module):
    def __init__(self, model_name="ViT-L/14", num_classes=1, unfreeze_blocks=4, local_model_path=None):
        super().__init__()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.backbone, _ = clip.load(local_model_path, device=device)
        self.backbone.float()  # ensure FP32
        hidden_size = self.backbone.visual.output_dim  # EyeCLIP final embedding size
        self.classifier = nn.Linear(hidden_size, num_classes).float()

        # Freeze backbone
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Unfreeze last N transformer blocks
        if unfreeze_blocks > 0:
            for block in self.backbone.visual.transformer.resblocks[-unfreeze_blocks:]:
                for param in block.parameters():
                    param.requires_grad = True

        # Always unfreeze classifier
        for param in self.classifier.parameters():
            param.requires_grad = True

    def forward(self, pixel_values):
        features = self.backbone.encode_image(pixel_values)
        return self.classifier(features)


# --------------------------
# Training loop (per fold, plain FP32)
# --------------------------
def train_one_fold(train_loader, val_loader, model, device, lr, epochs, num_classes):
    if num_classes == 1:
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    trainable_params = (p for p in model.parameters() if p.requires_grad)
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=100, T_mult=1)

    num_steps_per_epoch = len(train_loader)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for step, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            if num_classes == 1:
                labels = labels.float().unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            progress = epoch + step / num_steps_per_epoch
            scheduler.step(progress)

            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {running_loss/len(train_loader):.4f}")

    # --- Evaluation ---
    model.eval()
    y_true, y_pred, y_prob = [], [], []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)

            if num_classes == 1:
                probs = torch.sigmoid(outputs).squeeze(1)
                preds = (probs > 0.5).long()
                y_prob.extend(probs.cpu().numpy())
            else:
                probs = torch.softmax(outputs, dim=1)[:, 1]
                preds = torch.argmax(outputs, dim=1)
                y_prob.extend(probs.cpu().numpy())

            y_true.extend(labels.numpy())
            y_pred.extend(preds.cpu().numpy())

    return {
        "acc": accuracy_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_prob) if len(set(y_true)) > 1 else np.nan,
        "avg_prec": average_precision_score(y_true, y_prob) if len(set(y_true)) > 1 else np.nan,
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


# --------------------------
# Main StratifiedKFold CV
# --------------------------
def run_stratkfoldcv(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Collect dataset
    image_paths, labels = [], []
    for cls_name, cls_label in args.label_dict.items():
        cls_dir = os.path.join(args.data_path, cls_name)
        for f in os.listdir(cls_dir):
            if f.lower().endswith((".png", ".jpg", ".jpeg")):
                image_paths.append(os.path.join(cls_dir, f))
                labels.append(cls_label)

    image_paths, labels = np.array(image_paths), np.array(labels)

    # StratifiedKFold
    skf = StratifiedKFold(n_splits=args.num_folds, shuffle=True, random_state=42)

    results = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(image_paths, labels)):
        print(f"\n===== Fold {fold+1} / {args.num_folds} =====")
        model = EyeCLIPClassifier(args.model_name,
                                  num_classes=1 if len(set(labels)) == 2 else len(set(labels)),
                                  unfreeze_blocks=args.unfreeze_blocks,
                                  local_model_path=args.local_model_path).to(device)

        train_dataset = ImageDataset(image_paths[train_idx], labels[train_idx], augment=True)
        val_dataset   = ImageDataset(image_paths[val_idx], labels[val_idx], augment=False)

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

        metrics = train_one_fold(train_loader, val_loader, model, device, args.lr, args.epochs,
                                 1 if len(set(labels)) == 2 else len(set(labels)))
        metrics["fold"] = fold + 1
        results.append(metrics)
        print(metrics)

    # Save results
    df = pd.DataFrame(results)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    df.to_csv(os.path.join(args.output_dir, "per_fold_metrics.csv"), index=False)

    # Compute mean ± std
    summary_data = {}
    for metric in ["acc", "roc_auc", "avg_prec", "recall", "f1"]:
        mean_val = df[metric].mean()
        std_val = df[metric].std()
        summary_data[f"{metric}_mean"] = mean_val
        summary_data[f"{metric}_std"] = std_val
        summary_data[f"{metric}_mean±std"] = f"{mean_val:.4f} ± {std_val:.4f}"

    summary_df = pd.DataFrame([summary_data])
    summary_df.to_csv(os.path.join(args.output_dir, "summary_metrics.csv"), index=False)

    print("\n==== Per Fold Metrics ====")
    print(df.to_string(index=False))
    print("\n==== Final Mean ± Std Metrics ====")
    for metric in ["acc", "roc_auc", "avg_prec", "recall", "f1"]:
        print(f"{metric}: {summary_data[f'{metric}_mean±std']}")


# --------------------------
# CLI
# --------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True,
                        help="Root dir containing class subfolders (e.g., dr_class, nm_class)")
    parser.add_argument("--model_name", type=str, default="ViT-L/14")
    parser.add_argument("--local_model_path", type=str, default="/mnt/d/Naved/Codes/EyeCLIP/models")
    parser.add_argument("--output_dir", type=str, default="./results_eyeclip")
    parser.add_argument("--num_folds", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--label_dict", type=dict, default={"dr_class": 1, "nm_class": 0})
    parser.add_argument("--unfreeze_blocks", type=int, default=4,
                        help="Number of transformer blocks to unfreeze from the end of the vision encoder")
    args = parser.parse_args()

    run_stratkfoldcv(args)


"""
python eyeclip_finetune.py \
  --local_model_path /mnt/d/Naved/Codes/EyeCLIP/models/ViT-L-14.pt \
  --data_path /mnt/d/Naved/Data/IDRiD/idrid516_224x224 \
  --epochs 20 \
  --num_folds 10 \
  --unfreeze_blocks 2 \
  --output_dir /mnt/d/Naved/Outputs/idrid516_224x224/eyeclip_ft

"""