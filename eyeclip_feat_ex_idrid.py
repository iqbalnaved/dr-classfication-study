import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
import sys

# add EyeCLIP repo to path
# sys.path.append("/mnt/d/Naved/Codes/EyeCLIP")
import eyeclip.clip as clip

np.set_printoptions(threshold=np.inf)
np.random.seed(1)
torch.manual_seed(1)

# -------------------------
# Load EyeCLIP ViT-L/14
# -------------------------
def prepare_model(local_model_path=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model, preprocess = clip.load(local_model_path, device=device)
    
    model.eval()
    return model, preprocess, device


# -------------------------
# Process one image
# -------------------------
def run_one_image(img: Image.Image, preprocess, model, device):
    image_tensor = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = model.encode_image(image_tensor)  # CLIP-style feature
    return feat.squeeze().cpu().numpy()

# -------------------------
# Extract features for all images in folder
# -------------------------
def get_feature(data_path, preprocess, model, device, label_dict=None):
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
        latent_feature = run_one_image(img, preprocess, model, device)
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
if __name__ == "__main__":
    model, preprocess, device = prepare_model(
        # local_model_path= "/mnt/d/Naved/Codes/EyeCLIP/models/ViT-L-14.pt"  # directory with .pt weights
        local_model_path= "/mnt/d/Naved/Codes/EyeCLIP/models/ViT-L-14-336px.pt"  # directory with .pt weights
    )

    data_path = "/mnt/d/Naved/Data/IDRiD/idrid516_orig/"
    # save_path = "/mnt/d/Naved/Outputs/idrid516_orig/features/idrid516_orig_eyeclip_vitl14_features.csv"
    save_path = "/mnt/d/Naved/Outputs/idrid516_orig/features/idrid516_orig_eyeclip_vitl14_336px_features.csv"

    label_dict = {"dr": 1, "nm": 0}  # customize

    name_list, feature_list, label_list = get_feature(data_path, preprocess, model, device, label_dict)

    # os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df_feature = pd.DataFrame(feature_list)
    df_imgname = pd.DataFrame(name_list, columns=["name"])
    df_label = pd.DataFrame(label_list, columns=["label"])
    df_all = pd.concat([df_imgname, df_feature, df_label], axis=1)

    # rename feature columns
    feat_cols = [f"feature_{i}" for i in range(df_feature.shape[1])]
    df_all.columns = ["name"] + feat_cols + ["label"]

    df_all.to_csv(save_path, index=False)
    print("✅ Features saved to:", save_path)

