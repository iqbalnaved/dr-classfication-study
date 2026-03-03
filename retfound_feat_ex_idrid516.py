import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from PIL import Image
import models_vit as models
from huggingface_hub import hf_hub_download

np.set_printoptions(threshold=np.inf)
np.random.seed(1)
torch.manual_seed(1)


def prepare_model(chkpt_dir, arch='RETFound_mae'):
    
    # load model
    checkpoint = torch.load(chkpt_dir, map_location='cpu', weights_only=False)
    # print(checkpoint.keys())
    # build model
    if arch=='RETFound_mae':
        model = models.__dict__[arch](
            img_size=224,
            num_classes=5,
            drop_path_rate=0,
            global_pool=True,
        )
        msg = model.load_state_dict(checkpoint['model'], strict=False)
    elif arch=='RETFound_dinov2':
        model = models.__dict__['RETFound_mae'](
            img_size=224,
            num_classes=5,
            drop_path_rate=0,
            global_pool=True,
        )
        msg = model.load_state_dict(checkpoint['teacher'], strict=False)
    else:
        print(models.__dict__.keys())
        
        model = models.__dict__[arch](
            num_classes=5,
            drop_path_rate=0,
            args=None,
        )
        msg = model.load_state_dict(checkpoint['teacher'], strict=False)
    return model

def run_one_image(img, model, arch):
    
    x = torch.tensor(img)
    x = x.unsqueeze(dim=0)
    x = torch.einsum('nhwc->nchw', x)
    
    x = x.to(device, non_blocking=True)
    latent = model.forward_features(x.float())
    
    if arch=='dinov2_large':
        latent = latent[:, 1:, :].mean(dim=1,keepdim=True)
        latent = nn.LayerNorm(latent.shape[-1], eps=1e-6).to(device)(latent)
    
    latent = torch.squeeze(latent)

    return latent
    
def get_feature(data_path,
                chkpt_dir,
                device,
                arch='RETFound_mae',
                label_dict=None):
    # loading model
    model_ = prepare_model(chkpt_dir, arch)
    model_.to(device)

    # collect all image paths from subdirectories
    img_list = []
    for root, _, files in os.walk(data_path):
        for f in files:
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
                img_list.append(os.path.join(root, f))
    
    name_list = []
    feature_list = []
    label_list = []
    model_.eval()
    
    finished_num = 0
    for i in img_list:
        finished_num += 1
        if (finished_num % 1000 == 0):
            print(str(finished_num) + " finished")
        
        print(f"{finished_num}/{len(img_list)} extracting {i}")
        
        img = Image.open(i)  # i is already full path
        img = img.resize((224, 224))
        img = np.array(img) / 255.
        img[...,0] = (img[...,0] - img[...,0].mean())/img[...,0].std()
        img[...,1] = (img[...,1] - img[...,1].mean())/img[...,1].std()
        img[...,2] = (img[...,2] - img[...,2].mean())/img[...,2].std()
        assert img.shape == (224, 224, 3)
        
        latent_feature = run_one_image(img, model_, arch)
        
        name_list.append(i)
        feature_list.append(latent_feature.detach().cpu().numpy())
        
        # infer label from directory
        label_value = None
        if label_dict is not None:
            folder_name = os.path.basename(os.path.dirname(i)).lower()
            for key, val in label_dict.items():
                if key.lower() in folder_name:
                    label_value = val
                    break
        label_list.append(label_value)
        
    return [name_list, feature_list, label_list]


#------------

model_paths = [
               "RETFound_mae_natureCFP/RETFound_mae_natureCFP.pth", #3 feat size = 1x1024
               "RETFound_mae_natureOCT/RETFound_mae_natureOCT.pth" # 5
              ]

model_archs = ['RETFound_mae',               
               'RETFound_mae']

#------------
model_num = 0 # RETFound_mae_natureCFP.
model_filename = os.path.basename(model_paths[model_num]).split('.')[0]              
data_path = '/mnt/d/Naved/Data/IDRiD/idrid516_orig/'  # 'DATA_PATH'
save_path = '/mnt/d/Naved/Outputs/idrid516_orig/features/idrid516_orig_'+model_filename+'_features.csv'

#------------


chkpt_dir = "/mnt/d/Naved/Codes/RETFound_MAE/RETFound_hf_models/" + model_paths[model_num] 
#hf_hub_download(repo_id="YukunZhou/RETFound_dinov2_meh", filename="RETFound_dinov2_meh.pth")

arch = model_archs[model_num] # RETFound_mae, dinov2_large

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

label_dict = {'dr': 1, 'nm': 0}  # customize as needed

[name_list, feature, label_list] = get_feature(
    data_path,
    chkpt_dir,
    device,
    arch=arch,
    label_dict=label_dict
)

# ensure save directory exists
os.makedirs(os.path.dirname(save_path))

# save the feature
df_feature = pd.DataFrame(feature)
df_imgname = pd.DataFrame(name_list, columns=["name"])
df_label = pd.DataFrame(label_list, columns=["label"])
df_visualization = pd.concat([df_imgname, df_feature, df_label], axis=1)

# set feature column names
column_name_list = ["feature_{}".format(i) for i in range(df_feature.shape[1])]
df_visualization.columns = ["name"] + column_name_list + ["label"]

df_visualization.to_csv(save_path, index=False)

                