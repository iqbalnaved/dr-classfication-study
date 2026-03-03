import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import json

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, recall_score, f1_score

from transformers import AutoProcessor, AutoModel
from torch.amp import autocast, GradScaler
import torchvision.transforms as T
from PIL import Image


# --------------------------
# Dataset
# --------------------------
class ImageDataset(Dataset):
    def __init__(self, image_paths, labels, processor, augment=False):
        self.image_paths = image_paths
        self.labels = labels
        self.processor = processor
        if augment:
            self.transform = T.Compose([
                T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
                T.RandomApply([T.GaussianBlur(kernel_size=3)], p=0.5),
                T.RandomRotation(15),
            ])
        else:
            self.transform = None

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return inputs, label


# --------------------------
# Model wrapper (MedSigLIP encoder + classifier head)
# --------------------------
class MedSigLIPClassifier(nn.Module):
    def __init__(self, model_name, num_classes=1, unfreeze_blocks=4):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        hidden_size = self.backbone.vision_model.config.hidden_size
        self.classifier = nn.Linear(hidden_size, num_classes)

        # Freeze all backbone params
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Unfreeze last `unfreeze_blocks` transformer layers
        if unfreeze_blocks > 0:
            for block in self.backbone.vision_model.encoder.layers[-unfreeze_blocks:]:
                for param in block.parameters():
                    param.requires_grad = True

        # Always unfreeze classifier
        for param in self.classifier.parameters():
            param.requires_grad = True

    def forward(self, pixel_values):
        features = self.backbone.get_image_features(pixel_values=pixel_values)
        return self.classifier(features)


# --------------------------
# Training loop (per fold)
# --------------------------
def train_one_fold(train_loader, val_loader, model, device, lr, epochs, num_classes):
    if num_classes == 1:
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=100, T_mult=1)
    scaler = GradScaler(device="cuda")

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs = {k: v.to(device) for k, v in inputs.items()}
            labels = labels.to(device)

            if num_classes == 1:
                labels = labels.float().unsqueeze(1)

            optimizer.zero_grad()
            with autocast("cuda"):
                outputs = model(inputs["pixel_values"])
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step(epoch + len(train_loader))
            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {running_loss/len(train_loader):.4f}")

    # --- Evaluation ---
    model.eval()
    y_true, y_pred, y_prob = [], [], []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(inputs["pixel_values"])

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
    processor = AutoProcessor.from_pretrained(args.model_name, use_fast=True)
    if hasattr(processor, "image_processor"):
        processor.image_processor.do_resize = False


    # Load dataset paths
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
        train_dataset = ImageDataset(image_paths[train_idx], labels[train_idx], processor, augment=True)
        val_dataset = ImageDataset(image_paths[val_idx], labels[val_idx], processor, augment=False)

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

        num_classes = 1 if len(set(labels)) == 2 else len(set(labels))
        model = MedSigLIPClassifier(args.model_name, num_classes=num_classes,
                                    unfreeze_blocks=args.unfreeze_blocks).to(device)

        metrics = train_one_fold(train_loader, val_loader, model, device, args.lr, args.epochs, num_classes)
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
    parser.add_argument("--model_name", type=str, default="google/medsiglip-448")
    parser.add_argument("--output_dir", type=str, default="./results_medsiglip")
    parser.add_argument("--num_folds", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--label_dict", type=json.loads, default={"dr_class": 1, "nm_class": 0})
    parser.add_argument("--unfreeze_blocks", type=int, default=4,
                        help="Number of transformer blocks to unfreeze from the end of the vision encoder")
    args = parser.parse_args()

    run_stratkfoldcv(args)


"""

python medsiglip_ft.py \
    --data_path /mnt/d/Naved/Data/IDRiD/idrid516_448x448 \
    --model_name google/medsiglip-448 \
    --num_folds 10 \
    --epochs 20 \
    --batch_size 4 \
    --lr 5e-5 \
    --unfreeze_blocks 2 \
    --output_dir /mnt/d/Naved/Outputs/idrid516_448x448/medsiglip_ft


python medsiglip_ft.py \
    --data_path /mnt/d/Naved/Data/ISIC100/images/isic99_448 \
    --model_name google/medsiglip-448 \
    --num_folds 10 \
    --epochs 20 \
    --batch_size 4 \
    --lr 5e-5 \
    --unfreeze_blocks 2 \
    --output_dir /mnt/d/Naved/Outputs/isic99_448/medsiglip_ft \
    --label_dict '{"mm": 1, "bn": 0}'


"""