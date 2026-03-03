import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from transformers import AutoProcessor, AutoModel

from huggingface_hub import login
login("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")


np.set_printoptions(threshold=np.inf)
np.random.seed(1)
torch.manual_seed(1)

# -------------------------
# Load MedSigLIP-448 model
# -------------------------
def prepare_model(model_name="google/medsiglip-448"):
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    return processor, model

# -------------------------
# Process one image
# -------------------------
def run_one_image(img: Image.Image, processor, model, device):
    # Option A: rely on processor to resize and normalize for you
    inputs = processor(images=img, return_tensors="pt").to(device)
    # Option B (explicit resize) *if you want control*:
    # img_resized = img.resize((448, 448), resample=Image.BILINEAR)
    # inputs = processor(images=img_resized, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.get_image_features(**inputs)  # [batch=1, dim]
    latent = outputs.squeeze().cpu().numpy()
    return latent

# -------------------------
# Extract features for all images in folder
# -------------------------
def get_feature(data_path, processor, model, device, label_dict=None):
    img_list = []
    for root, _, files in os.walk(data_path):
        for f in files:
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
                img_list.append(os.path.join(root, f))

    name_list, feature_list, label_list = [], [], []
    model.eval()

    for idx, path in enumerate(img_list, 1):
        print(f"{idx}/{len(img_list)} processing {path}")
        img = Image.open(path).convert("RGB")
        latent_feature = run_one_image(img, processor, model, device)
        name_list.append(path)
        feature_list.append(latent_feature)

        label_value = None
        if label_dict is not None:
            folder_name = os.path.basename(os.path.dirname(path)).lower()
            for key, val in label_dict.items():
                if key.lower() in folder_name:
                    label_value = val
                    break
        label_list.append(label_value)

    return name_list, feature_list, label_list

# -------------------------
# Main
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

processor, model = prepare_model("google/medsiglip-448") # 1x1024 size features
model.to(device)

# data_path = "/mnt/d/Naved/Data/IDRiD/idrid516_orig/"
# save_path = "/mnt/d/Naved/Outputs/idrid516_orig/features/idrid516_orig_medsiglip448_features.csv"
# label_dict = {"dr": 1, "nm": 0}  # customize

# data_path = "/mnt/d/Naved/Data/ISIC99/images/originals/"
# save_path = "/mnt/d/Naved/Outputs/isic99_orig/features/isic99_orig_medsiglip448_features.csv"
# label_dict = {"mm": 1, "bn": 0}  # customize

data_path = "/mnt/d/Naved/Data/MRA-MIDAS/midas271"
save_path = "/mnt/d/Naved/Outputs/midas271_orig/features/midas271_orig_medsiglip448_features.csv"
label_dict = {"mm": 1, "bn": 0}  # customize

name_list, feature_list, label_list = get_feature(data_path, processor, model, device, label_dict)

os.makedirs(os.path.dirname(save_path), exist_ok=True)
df_feature = pd.DataFrame(feature_list)
df_imgname = pd.DataFrame(name_list, columns=["name"])
df_label = pd.DataFrame(label_list, columns=["label"])
df_all = pd.concat([df_imgname, df_feature, df_label], axis=1)

# column names for features
feat_cols = [f"feature_{i}" for i in range(df_feature.shape[1])]
df_all.columns = ["name"] + feat_cols + ["label"]

df_all.to_csv(save_path, index=False)
print("✅ Features saved to:", save_path)
