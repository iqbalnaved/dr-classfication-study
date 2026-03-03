# k-fold CV with a single CSV file of RETFound features
import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, recall_score, f1_score
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# -----------------------------
# Inputs
# -----------------------------

# path to your single CSV file (features + filename + label)
# csv_path = "/mnt/d/Naved/Outputs/idrid516_orig/features/idrid516_orig_RETFound_mae_natureCFP_features.csv"
# save_path = "/mnt/d/Naved/Outputs/idrid516_orig/preds"
# dbname = 'IDRiD516_orig'
# pretrained_model_name = 'RETFound_mae_natureCFP'

# csv_path = "/mnt/d/Naved/Outputs/idrid516_orig/features/idrid516_orig_medsiglip448_features.csv"
# save_path = "/mnt/d/Naved/Outputs/idrid516_orig/medsiglip448_preds"
# dbname = 'IDRiD516_orig'
# pretrained_model_name = 'google_medsiglip448'

# csv_path = "/mnt/d/Naved/Outputs/idrid516_orig/features/idrid516_orig_vitlarge224_features.csv"
# save_path = "/mnt/d/Naved/Outputs/idrid516_orig/vitlarge224_preds"
# dbname = 'IDRiD516_orig'
# pretrained_model_name = 'vitlarge224'

# csv_path = "/mnt/d/Naved/Outputs/idrid516_orig/features/idrid516_orig_eyeclip_vitl14_features.csv"
# save_path = "/mnt/d/Naved/Outputs/idrid516_orig/eyeclip_vitl14_preds"
# dbname = 'IDRiD516_orig'
# pretrained_model_name = 'eyeclip_vitl14'

# csv_path = "/mnt/d/Naved/Outputs/idrid516_orig/features/idrid516_orig_eyeclip_vitl14_336px_features.csv"
# save_path = "/mnt/d/Naved/Outputs/idrid516_orig/eyeclip_vitl14_336px_preds"
# dbname = 'IDRiD516_orig'
# pretrained_model_name = 'eyeclip_vitl14_336px'

# csv_path = "/mnt/d/Naved/Outputs/isic99_orig/features/isic99_orig_medsiglip448_features.csv"
# save_path = "/mnt/d/Naved/Outputs/idrid99_orig/isic99_medsiglip448_preds"
# dbname = 'ISIC99_orig'
# pretrained_model_name = 'medsiglip448'

csv_path = "/mnt/d/Naved/Outputs/midas271_orig/features/midas271_orig_medsiglip448_features.csv"
save_path = "/mnt/d/Naved/Outputs/midas271_orig/midas271_orig_medsiglip448_preds"
dbname = 'MIDAS271_orig'
pretrained_model_name = 'medsiglip448'

if not os.path.isdir(save_path):
    os.makedirs(save_path)

# -----------------------------
# Load CSV
# -----------------------------
df = pd.read_csv(csv_path)

# assume first column = filename, last column = label, middle = features
X = df.iloc[:, 1:-1].values
y = df.iloc[:, -1].values

# Encode labels if not numeric
le = LabelEncoder()
y = le.fit_transform(y)

# -----------------------------
# Models
# -----------------------------
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
}

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# -----------------------------
# Cross-validation (all metrics per fold)
# -----------------------------
all_fold_results = []   # detailed results
summary_results = []    # aggregated results

for name, model in models.items():
    print(f'\n{dbname}: {name} Cross-Validation Results:')

    acc_scores, roc_scores, ap_scores, rec_scores, f1_scores = [], [], [], [], []

    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y), 1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

        acc = accuracy_score(y_test, y_pred)
        ap = average_precision_score(y_test, y_prob) if y_prob is not None else np.nan
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc = roc_auc_score(y_test, y_prob) if y_prob is not None else np.nan

        # collect per-fold metrics
        all_fold_results.append({
            "Model": name,
            "Fold": fold,
            "Accuracy": acc,
            "ROC-AUC": roc,
            "Avg Precision": ap,
            "Recall": rec,
            "F1": f1
        })

        acc_scores.append(acc)
        rec_scores.append(rec)
        f1_scores.append(f1)
        if not np.isnan(roc):
            roc_scores.append(roc)
        if not np.isnan(ap):
            ap_scores.append(ap)            

    # summary stats
    summary_results.append({
        "Model": name,
        "Accuracy": f"{np.mean(acc_scores):.4f} ± {np.std(acc_scores):.4f}",
        "ROC-AUC": f"{np.mean(roc_scores):.4f} ± {np.std(roc_scores):.4f}" if roc_scores else "N/A",
        "Avg Precision": f"{np.mean(ap_scores):.4f} ± {np.std(ap_scores):.4f}" if ap_scores else "N/A",
        "Recall": f"{np.mean(rec_scores):.4f} ± {np.std(rec_scores):.4f}",
        "F1": f"{np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}"
    })

    # print nicely
    print(f"Accuracy : {np.mean(acc_scores):.4f} ± {np.std(acc_scores):.4f}")
    if roc_scores:
        print(f"ROC-AUC  : {np.mean(roc_scores):.4f} ± {np.std(roc_scores):.4f}")
    if ap_scores:
        print(f"Avg Precision: {np.mean(ap_scores):.4f} ± {np.std(ap_scores):.4f}")    
    print(f"Recall   : {np.mean(rec_scores):.4f} ± {np.std(rec_scores):.4f}")
    print(f"F1 Score : {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")

# -----------------------------
# Save results
# -----------------------------
fold_csv = os.path.join(save_path, f"{pretrained_model_name}_fold_metrics.csv")
summary_csv = os.path.join(save_path, f"{pretrained_model_name}_summary_metrics.csv")

pd.DataFrame(all_fold_results).to_csv(fold_csv, index=False)
pd.DataFrame(summary_results).to_csv(summary_csv, index=False)

print(f"\n✅ Per-fold metrics saved to {fold_csv}")
print(f"✅ Summary metrics saved to {summary_csv}")