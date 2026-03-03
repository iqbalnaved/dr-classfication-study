import os
import re
import json
import argparse
from datetime import datetime
import random

import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
from huggingface_hub import login


hf_token = os.getenv("HF_TOKEN")
if hf_token:
    login(hf_token)



def load_model(model_id: str = "google/medgemma-4b-it"): # google/medgemma-1.5-4b-it
    if torch.cuda.is_available():
        device = "cuda"
        gpu_props = torch.cuda.get_device_properties(0)
        vram_gb = gpu_props.total_memory / 1024 ** 3

        if vram_gb < 12:
            bnb_cfg = BitsAndBytesConfig(load_in_4bit=True)
            model = AutoModelForImageTextToText.from_pretrained(
                model_id,
                device_map="auto",
                dtype=torch.bfloat16,
                offload_buffers=True,
                quantization_config=bnb_cfg,
            )
        else:
            model = AutoModelForImageTextToText.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,
                device_map={"": "cuda"},
            )
    else:
        device = "cpu"
        model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            device_map=None,
        )

    processor = AutoProcessor.from_pretrained(model_id, use_fast=True)

    if not any(p.device.type == "cuda" for p in model.parameters()):
        device = "cpu"

    return model, processor, device



def build_fewshot_messages(support_set, query_image, question):

    instruction = (
        "You are an ophthalmologist specializing in retinal fundus analysis and diabetic "
        "retinopathy (DR).\n\n"
        "You MUST respond with STRICT JSON only in the following format:\n\n"
        "{\n"
        '  "diagnosis": "DR" or "Normal",\n'
        '  "rationale": "your rationale for this diagnosis",\n'
        '  "confidence": number between 0 and 1\n'
        "}\n\n"
        "Do not include any additional text."
    )

    messages = []

    support_text = {
        "DR": json.dumps({
            "diagnosis": "DR",
            "rationale": "Presence of microaneurysms, hemorrhages, or exudates.",
            "confidence": 0.95
        }),
        "Normal": json.dumps({
            "diagnosis": "Normal",
            "rationale": "No visible pathological signs.",
            "confidence": 0.95
        })
    }

    # ---- few-shot examples ----
    for img, label in support_set:
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": instruction},
                {"type": "image", "image": img},
                {"type": "text", "text": question},
            ],
        })

        messages.append({
            "role": "assistant",
            "content": [{"type": "text", "text": support_text[label]}],
        })

    # ---- query ----
    messages.append({
        "role": "user",
        "content": [
            {"type": "text", "text": instruction},
            {"type": "image", "image": query_image},
            {"type": "text", "text": question},
        ],
    })

    return messages




def analyze_image_multimodal(query_path, support_paths, question, model, processor, device, max_new_tokens=250):

    query_img = Image.open(query_path).convert("RGB")

    # load support images
    support = [(Image.open(p).convert("RGB"), lbl) for p, lbl in support_paths]

    messages = build_fewshot_messages(support, query_img, question)

    # collect support + query images
    support_imgs = [x[0] for x in support]
    images = support_imgs + [query_img]

    prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = processor(images=images, text=prompt, return_tensors="pt")

    target_device = next(model.parameters()).device

    for k, v in inputs.items():
        if k == "input_ids":
            inputs[k] = v.to(target_device, dtype=torch.long)
        else:
            inputs[k] = v.to(target_device, dtype=model.dtype)

    input_length = inputs["input_ids"].shape[-1]

    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        return_dict_in_generate=True,
        do_sample=False, #True,
        # temperature=0.7,
        # top_p=0.9
    )

    sequences = output.sequences
    generated_ids = sequences[0, input_length:] #if sequences.ndim == 2 else sequences[input_length:]
    decoded = processor.decode(generated_ids, skip_special_tokens=True).strip()

    return decoded



def _extract_json_block(text: str):
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start:end + 1]
    return None


def _parse_confidence_from_text(text: str):
    nums = re.findall(r"\b\d+(?:\.\d+)?\b", text)
    for raw in nums:
        try:
            val = float(raw)
            if 1 < val <= 100:
                return max(0.0, min(1.0, val / 100.0))
            if 0 <= val <= 1:
                return val
        except:
            continue
    return None



def parse_model_response(text: str):
    raw = text or ""
    block = _extract_json_block(raw)

    def normalize_diag(s: str):
        s = (s or "").strip().lower()

        # strong "normal" / "no DR" signals
        normal_markers = [
            "normal", "no dr", "no diabetic retinopathy", "without diabetic retinopathy",
            "no evidence of diabetic retinopathy", "healthy"
        ]
        if any(m in s for m in normal_markers):
            return "normal"

        # strong DR signals
        dr_markers = [
            "diabetic retinopathy", "retinopathy", "npdr", "pdr",
            "microaneurysm", "hemorrhage", "exudate"
        ]
        if any(m in s for m in dr_markers):
            return "dr"

        # fallback: literal dr token (as a standalone token helps)
        if re.search(r"\bdr\b", s):
            return "dr"

        return "error"

    if not block:
        diagnosis = normalize_diag(raw)
        return diagnosis, raw, _parse_confidence_from_text(raw)

    try:
        obj = json.loads(block)
    except Exception:
        diagnosis = normalize_diag(raw)
        return diagnosis, raw, _parse_confidence_from_text(raw)

    diag_text = str(obj.get("diagnosis", ""))
    diagnosis = normalize_diag(diag_text)

    conf = obj.get("confidence", None)
    try:
        conf = float(conf)
        if 1 < conf <= 100:
            conf /= 100.0
        conf = max(0.0, min(1.0, conf))
    except Exception:
        conf = _parse_confidence_from_text(raw)

    rationale = obj.get("rationale", raw)
    return diagnosis, rationale, conf




def compute_metrics(df: pd.DataFrame):
    TP = ((df.true_label == "dr") & (df.prediction == "dr")).sum()
    TN = ((df.true_label == "normal") & (df.prediction == "normal")).sum()
    FP = ((df.true_label == "normal") & (df.prediction == "dr")).sum()
    FN = ((df.true_label == "dr") & (df.prediction == "normal")).sum()

    total = len(df)
    accuracy = (TP + TN) / total if total else 0
    sensitivity = TP / (TP + FN) if (TP + FN) else 0
    specificity = TN / (TN + FP) if (TN + FP) else 0

    return accuracy, sensitivity, specificity



def process_dataset(root_dir, output_dir, run_number, model_id, k):

    os.makedirs(output_dir, exist_ok=True)

    model, processor, device = load_model(model_id)
    print(f"Using device: {device}")

    question = (
        "Based on this retinal fundus image, is the diagnosis Diabetic Retinopathy (DR) "
        "or Normal? Respond ONLY in JSON. Diagnosis must be exactly DR or Normal (no other words)."
    )

    classes = {"dr_class": "DR", "nm_class": "Normal"}

    # collect all file paths per class
    all_files = {}
    for folder in classes.keys():
        subdir = os.path.join(root_dir, folder)
        all_files[folder] = [
            os.path.join(subdir, f)
            for f in os.listdir(subdir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]

    rows = []

    for folder, label in classes.items():
        for query_path in tqdm(all_files[folder], desc=f"Evaluating {folder}"):

            # ---- resample support per query, WITH replacement, excluding the query ----
            support_paths = []

            if k > 0: 
                for target_folder, target_label in classes.items():

                    file_list = all_files[target_folder]

                    # Exclude the query image from eligible support
                    eligible = [p for p in file_list if p != query_path]

                    if not eligible:  # corner case → use full list
                        eligible = file_list

                    chosen = random.choices(eligible, k=k)
                    support_paths.extend([(p, target_label) for p in chosen])

            try:
                raw = analyze_image_multimodal(query_path, support_paths, question, model, processor, device)
                pred, rationale, conf = parse_model_response(raw)
                
                # ADDED PRINTS (same behavior as zero-shot)
                print(f"\n🖼️ Image: {os.path.basename(query_path)}")
                print(f"📘 Raw / rationale: {raw}")
                print(f"🔎 Prediction: {pred}, true label: {label.lower()}")
                print(f"📊 Confidence: {conf}")
    
            except Exception as e:
                raw = str(e)
                pred, rationale, conf = "error", raw, None

            rows.append(
                {
                    "image": os.path.basename(query_path),
                    "true_label": label.lower(),
                    "prediction": pred,
                    "rationale": rationale,
                    "confidence": conf,
                    "raw_response": raw,
                }
            )

    df = pd.DataFrame(rows)
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    tag = f"run{run_number}"

    prefix = "zeroshot" if k == 0 else "fewshot_dynamic_excl"
    json_path = os.path.join(output_dir, f"{prefix}_{tag}_{ts}.json")

    df.to_json(json_path, orient="records", indent=2)
    print(f"Saved results: {json_path}")

    acc, sens, spec = compute_metrics(df)
    print(f"Accuracy: {acc:.4f}")
    print(f"Sensitivity: {sens:.4f}")
    print(f"Specificity: {spec:.4f}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()    
    parser.add_argument("root_dir")
    parser.add_argument("output_dir")
    parser.add_argument("run_number", type=int)
    parser.add_argument("--m", type=str, default="google/medgemma-4b-it")
    parser.add_argument("--k", type=int, required=True)

    args = parser.parse_args()

    process_dataset(args.root_dir, args.output_dir, args.run_number, args.m, args.k)

""""
python medgemma_kshot_idrid.py /mnt/d/Naved/Data/IDRiD/idrid516_orig/ /mnt/d/Naved/Outputs/idrid516_orig/medgemma_kshot/ 1 --k 0
python medgemma_kshot_idrid.py /mnt/d/Naved/Data/IDRiD/idrid516_orig/ /mnt/d/Naved/Outputs/idrid516_orig/medgemma_kshot/ 2 --k 0
python medgemma_kshot_idrid.py /mnt/d/Naved/Data/IDRiD/idrid516_orig/ /mnt/d/Naved/Outputs/idrid516_orig/medgemma_kshot/ 3 --k 0

python medgemma_kshot_idrid.py /mnt/d/Naved/Data/IDRiD/idrid516_orig/ /mnt/d/Naved/Outputs/idrid516_orig/medgemma_kshot/ 1 --k 5
python medgemma_kshot_idrid.py /mnt/d/Naved/Data/IDRiD/idrid516_orig/ /mnt/d/Naved/Outputs/idrid516_orig/medgemma_kshot/ 2 --k 5
python medgemma_kshot_idrid.py /mnt/d/Naved/Data/IDRiD/idrid516_orig/ /mnt/d/Naved/Outputs/idrid516_orig/medgemma_kshot/ 3 --k 5

--
python medgemma_kshot_idrid.py /mnt/d/Naved/Data/IDRiD/idrid516_orig/ /mnt/d/Naved/Outputs/idrid516_orig/medgemma1.5_kshot/ 1 --m google/medgemma-1.5-4b-it  --k 0
python medgemma_kshot_idrid.py /mnt/d/Naved/Data/IDRiD/idrid516_orig/ /mnt/d/Naved/Outputs/idrid516_orig/medgemma1.5_kshot/ 2 --m google/medgemma-1.5-4b-it  --k 0
python medgemma_kshot_idrid.py /mnt/d/Naved/Data/IDRiD/idrid516_orig/ /mnt/d/Naved/Outputs/idrid516_orig/medgemma1.5_kshot/ 3 --m google/medgemma-1.5-4b-it  --k 0

python medgemma_kshot_idrid.py /mnt/d/Naved/Data/IDRiD/idrid516_orig/ /mnt/d/Naved/Outputs/idrid516_orig/medgemma1.5_kshot/ 1 --m google/medgemma-1.5-4b-it  --k 5
python medgemma_kshot_idrid.py /mnt/d/Naved/Data/IDRiD/idrid516_orig/ /mnt/d/Naved/Outputs/idrid516_orig/medgemma1.5_kshot/ 2 --m google/medgemma-1.5-4b-it  --k 5
python medgemma_kshot_idrid.py /mnt/d/Naved/Data/IDRiD/idrid516_orig/ /mnt/d/Naved/Outputs/idrid516_orig/medgemma1.5_kshot/ 3 --m google/medgemma-1.5-4b-it  --k 5
"""