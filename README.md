# [Comparative Analysis of General-Purpose vs. Domain-Specific Multimodal Models for Diabetic Retinopathy Classification]

**Paper link:** [PAPER LINK / DOI PLACEHOLDER]

**Main diagram / overview figure:**  
[diagram.png]

---

## Dataset

**Database:** IDRiD (DR:348, NR:168)  
**Image size:** 4288x2848  
**Source:** https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid

---

## Directory Structure

```text
<parent-dir>/Data/IDRiD/idrid516_orig/dr_class
<parent-dir>/Data/IDRiD/idrid516_orig/nm_class
```

---

## API Setup

### Environment Variables

Set the following keys in your environment:

- OPENAI_API_KEY
- GEMINI_API_KEY
- MISTRAL_API_KEY

### Bash Setup

```bash
nano .bashrc
```

```bash
export OPENAI_API_KEY='<API-KEY>'
```

Save and close, then apply changes:

```bash
source ~/.bashrc
```

---

## Third-Party Codes and Models

- RETFound (official repo): https://github.com/openmedlab/RETFound_MAE
- EyeCLIP (official repo): https://github.com/Michi-3000/EyeCLIP
- MedGemma:
  - https://huggingface.co/google/medgemma-4b-it
  - https://huggingface.co/google/medgemma-1.5-4b-it
- MedSigLIP: https://huggingface.co/google/medsiglip-448

---

## Zero- and Few-Shot Runs

### GPT-Based Runs

```bash
python3 idrid_gpt.py -s 0 -r 1 -m gpt-5.2-2025-12-11 -d IDRiD516_orig -k 0
```

### Gemini-Based Runs

```bash
python3 idrid_gemini_resume.py --shot 0 --run 1 --model gemini-3-pro-preview --dataset IDRiD516_orig -k 0
```

### Mistral / Pixtral Runs

```bash
python3 idrid_mistral.py -s 0 -r 1 -m pixtral-large-2411 -d IDRiD516_orig -k 0
```

### MedGemma K-Shot Runs

```bash
python medgemma_kshot_idrid.py /mnt/d/Naved/Data/IDRiD/idrid516_orig/ /mnt/d/Naved/Outputs/idrid516_orig/medgemma_kshot/ 1 --k 0
```

```bash
python medgemma_kshot_idrid.py /mnt/d/Naved/Data/IDRiD/idrid516_orig/ /mnt/d/Naved/Outputs/idrid516_orig/medgemma1.5_kshot/ 1 --m google/medgemma-1.5-4b-it --k 5
```

---

## Linear Probe

- RETFound_MAE/retfound_feat_ex_idrid516.py
- medseglip_feat_ex.py
- EyeCLIP/eyeclip_feat_ex_idrid.py
- vit_large16_feat_ex_idrid.py
- lineprobe.py

---

## Fine Tuning

### RETFound

```bash
python retfound_finetune_idrid516.py   --data_path /mnt/d/Naved/Data/IDRiD/idrid516_224x224   --nb_classes 2   --epochs 20   --num_folds 10   --unfreeze_blocks 2   --output_dir /mnt/d/Naved/Outputs/idrid516_224x224/retfound_ft
```

### MedSigLIP

```bash
python medsiglip_ft.py   --data_path /mnt/d/Naved/Data/ISIC100/images/isic99_448   --model_name google/medsiglip-448   --num_folds 10   --epochs 20   --batch_size 4   --lr 5e-5   --unfreeze_blocks 2   --output_dir /mnt/d/Naved/Outputs/isic99_448/medsiglip_ft   --label_dict '{"mm": 1, "bn": 0}'
```

### EyeCLIP

```bash
python eyeclip_finetune.py   --local_model_path /mnt/d/Naved/Codes/EyeCLIP/models/ViT-L-14.pt   --data_path /mnt/d/Naved/Data/IDRiD/idrid516_224x224   --epochs 20   --num_folds 10   --unfreeze_blocks 2   --output_dir /mnt/d/Naved/Outputs/idrid516_224x224/eyeclip_ft
```

### ViT

```bash
python vit_ft.py   --data_path /mnt/d/Naved/Data/IDRiD/idrid516_224x224   --model_name google/vit-large-patch16-224   --num_folds 10   --epochs 30   --batch_size 8   --lr 1e-4   --unfreeze_blocks 2   --nb_classes 2   --pooling cls   --output_dir /mnt/d/Naved/Outputs/idrid516_224x224/vitlarge224_ft
```
