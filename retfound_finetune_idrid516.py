#!/usr/bin/env python3
"""
finetune_launcher.py

Stratified K-Fold Cross-Validation training & evaluation for RETFound/ViT models.
Computes Accuracy, ROC-AUC, Avg Precision, Recall, F1.
"""

import argparse
import os
from pathlib import Path
import numpy as np
import pandas as pd
import time

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    average_precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
from torch.amp import autocast, GradScaler

import models_vit

torch.backends.cudnn.benchmark = True


def run_stratified_kfold_cv(args):
    print(f"[Cross-Validation] Running StratifiedKFold with {args.num_folds} folds")

    # --- Data transforms with augmentation ---
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.ColorJitter(
            brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1
        ),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.5),
        transforms.RandomRotation(15),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


    full_dataset = datasets.ImageFolder(args.data_path, transform=val_transform)
    X = np.arange(len(full_dataset))
    y = full_dataset.targets
    n_classes = args.nb_classes

    skf = StratifiedKFold(n_splits=args.num_folds, shuffle=True, random_state=args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    metrics_per_fold = []
    start_total = time.time()   # Track full run time

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        fold_start = time.time()
        print(f"\n===== Fold {fold+1}/{args.num_folds} =====")

        # Subsets
        train_subset = Subset(datasets.ImageFolder(args.data_path, transform=train_transform), train_idx)
        val_subset   = Subset(datasets.ImageFolder(args.data_path, transform=val_transform), val_idx)

        train_loader = DataLoader(train_subset, batch_size=args.batch_size,
                                  shuffle=True, num_workers=args.workers,
                                  pin_memory=True)
        val_loader = DataLoader(val_subset, batch_size=args.batch_size,
                                shuffle=False, num_workers=args.workers,
                                pin_memory=True)

        # --- Model ---
        if n_classes == 2:
            model = getattr(models_vit, args.model)(num_classes=1, global_pool=args.global_pool)
            criterion = torch.nn.BCEWithLogitsLoss()
        else:
            model = getattr(models_vit, args.model)(num_classes=n_classes, global_pool=args.global_pool)
            criterion = torch.nn.CrossEntropyLoss()
        
        model.to(device)

        # === Partial fine-tuning control ===
        # Freeze everything first
        for p in model.parameters():
            p.requires_grad = False

        # Always unfreeze classifier head
        if hasattr(model, "head"):
            for p in model.head.parameters():
                p.requires_grad = True
        elif hasattr(model, "fc"):  # in case model uses .fc
            for p in model.fc.parameters():
                p.requires_grad = True

        # Locate transformer blocks (RETFound uses ViT backbone)
        blocks = None
        for cand in ["blocks", "encoder.blocks", "transformer.blocks"]:
            node = model
            try:
                for name in cand.split("."):
                    node = getattr(node, name)
                blocks = node
                break
            except AttributeError:
                pass

        assert blocks is not None, "❌ Could not locate transformer blocks in RETFound model."

        # Unfreeze policy
        if args.unfreeze_blocks < 0:
            # Full fine-tune (all layers)
            for p in model.parameters():
                p.requires_grad = True
        elif args.unfreeze_blocks > 0:
            for blk in list(blocks)[-args.unfreeze_blocks:]:
                for p in blk.parameters():
                    p.requires_grad = True
        # else: 0 = linear probe (only head trainable)

        # --- Optimizer with layer decay ---
        def param_groups(model, base_lr, layer_decay=0.75):
            groups = []
            layers = list(model.children())
            n = len(layers)
            for i, layer in enumerate(layers):
                scale = layer_decay ** (n - i - 1)
                groups.append({
                    "params": layer.parameters(),
                    "lr": base_lr * scale
                })
            return groups

        effective_lr = args.base_lr * (args.batch_size * args.accum_steps / 256.0)
        optimizer = torch.optim.AdamW(
            param_groups(model, effective_lr, args.layer_decay),
            weight_decay=args.weight_decay
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=100, T_mult=1
        )

        # --- Loss ---
        # if n_classes == 2:
            # criterion = torch.nn.BCEWithLogitsLoss()
        # else:
            # criterion = torch.nn.CrossEntropyLoss()

        best_val_loss = float("inf")
        best_state = None
        scaler = GradScaler(device="cuda")  # <-- One scaler per fold

        # --- Training ---
        for epoch in range(args.epochs):
            epoch_start = time.time()
            model.train()
            running_loss = 0.0

            for step, (imgs, labels) in enumerate(train_loader):
                imgs, labels = imgs.to(device), labels.to(device)
                if n_classes == 2:
                    labels = labels.float().unsqueeze(1)

                optimizer.zero_grad()
                with autocast("cuda"):  # AMP forward + loss
                    outputs = model(imgs)
                    loss = criterion(outputs, labels)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step(step + epoch * len(train_loader))
                running_loss += loss.item()

            # Validation loss
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for imgs, labels in val_loader:
                    imgs, labels = imgs.to(device), labels.to(device)
                    if n_classes == 2:
                        labels = labels.float().unsqueeze(1)
                    outputs = model(imgs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()

            avg_train_loss = running_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            epoch_time = time.time() - epoch_start
            print(f"Fold {fold+1} Epoch {epoch+1}/{args.epochs} "
                  f"- Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} "
                  f"| Time: {epoch_time:.2f}s")

            # Save best checkpoint
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_state = model.state_dict().copy()

        fold_time = time.time() - fold_start
        print(f"⏱️ Fold {fold+1} completed in {fold_time/60:.2f} min")

        # --- Evaluation with best checkpoint ---
        model.load_state_dict(best_state)
        all_preds, all_labels, all_probs = [], [], []
        model.eval()
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                if n_classes == 2:
                    probs = torch.sigmoid(outputs)
                    preds = (probs > 0.5).long()
                    all_probs.extend(probs.cpu().numpy())
                else:
                    probs = torch.softmax(outputs, dim=1)
                    preds = torch.argmax(probs, dim=1)
                    all_probs.extend(probs.cpu().numpy())

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        all_labels = np.array(all_labels)
        all_preds = np.array(all_preds)
        all_probs = np.array(all_probs)

        # Metrics
        acc = accuracy_score(all_labels, all_preds)
        try:
            if n_classes == 2:
                rocauc = roc_auc_score(all_labels, all_probs)
                avg_prec = average_precision_score(all_labels, all_probs)
            else:
                rocauc = roc_auc_score(all_labels, all_probs, multi_class="ovr", average="macro")
                avg_prec = average_precision_score(all_labels, all_probs, average="macro")
        except ValueError:
            rocauc, avg_prec = np.nan, np.nan

        recall = recall_score(all_labels, all_preds, average="macro", zero_division=0)
        f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

        cm = confusion_matrix(all_labels, all_preds)
        print(f"Fold {fold+1} Accuracy: {acc*100:.2f}% | ROC-AUC: {rocauc:.4f} | "
              f"AvgPrec: {avg_prec:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")
        print("Confusion Matrix:\n", cm)

        metrics_per_fold.append({
            "Fold": fold+1,
            "Accuracy": acc,
            "ROC-AUC": rocauc,
            "AvgPrecision": avg_prec,
            "Recall": recall,
            "F1": f1
        })

    # --- Summary ---
    df = pd.DataFrame(metrics_per_fold)
    summary = {
        "Fold": "Mean ± Std",
        "Accuracy": f"{df['Accuracy'].mean():.4f} ± {df['Accuracy'].std():.4f}",
        "ROC-AUC": f"{df['ROC-AUC'].mean():.4f} ± {df['ROC-AUC'].std():.4f}",
        "AvgPrecision": f"{df['AvgPrecision'].mean():.4f} ± {df['AvgPrecision'].std():.4f}",
        "Recall": f"{df['Recall'].mean():.4f} ± {df['Recall'].std():.4f}",
        "F1": f"{df['F1'].mean():.4f} ± {df['F1'].std():.4f}"
    }
    df = pd.concat([df, pd.DataFrame([summary])], ignore_index=True)

    print("\n===== Cross-Validation Results =====")
    print(df)

    out_path = Path(args.output_dir)
    if not out_path.exists():
        out_path.mkdir(parents=True)
    df.to_csv(out_path / "cv_results.csv", index=False)
    print(f"Saved per-fold + summary metrics to {out_path/'cv_results.csv'}")

    total_time = time.time() - start_total
    print(f"\n🏁 Total cross-validation time: {total_time/60:.2f} min ({total_time/3600:.2f} h)")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--nb_classes", type=int, required=True)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--num_folds", type=int, default=5)
    p.add_argument("--output_dir", type=str, default="./cv_outputs")
    # set params
    p.add_argument("--model", type=str, default="RETFound_mae")
    p.add_argument("--input_size", type=int, default=224)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--accum_steps", type=int, default=2)
    p.add_argument("--base_lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--layer_decay", type=float, default=0.75)
    p.add_argument("--global_pool", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--unfreeze_blocks", type=int, default=2,
               help="Unfreeze last N transformer blocks (0 = linear probe, negative = full FT)")

    return p.parse_args()


def main():
    args = parse_args()
    run_stratified_kfold_cv(args)


if __name__ == "__main__":
    main()


"""
python retfound_finetune_idrid516.py \
  --data_path /mnt/d/Naved/Data/IDRiD/idrid516_224x224 \
  --nb_classes 2 \
  --epochs 20 \
  --num_folds 10 \
  --unfreeze_blocks 2 \
  --output_dir /mnt/d/Naved/Outputs/idrid516_224x224/retfound_ft

"""