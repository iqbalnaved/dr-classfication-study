# ============================================================
# idrid_mistral.py
# Fully patched version of idrid_gpt.py for Mistral / Pixtral
# ============================================================

import os
import random
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from PIL import Image
import base64
import json
import argparse
import re
import pandas as pd
import time
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

from mistralai import Mistral

# -----------------------------
# API keys (rotate if needed)
# -----------------------------
keychain = [
    os.environ.get("MISTRAL_API_KEY1"),
    os.environ.get("MISTRAL_API_KEY2"),
    os.environ.get("MISTRAL_API_KEY3"),
]

# -----------------------------
# Pricing (USD / 1K tokens)
# Adjust to your contract
# -----------------------------
PRICING = {
    "pixtral-large-2411": {
        "input": 0.001,
        "output": 0.003
    },
    "pixtral-12b": {
        "input": 0.00015,
        "output": 0.00015
    }
}


# -----------------------------
# Utility functions
# -----------------------------

def encode_image_base64(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def clean_response(resp):
    cleaned = re.sub(r"^```(?:json)?", "", resp.strip())
    cleaned = re.sub(r"```$", "", cleaned)
    return cleaned.strip()


def sanitize(text):
    return re.sub(r'[\x00-\x1F\x7F]', '', text)


def calculate_chat_cost(response):
    """
    Robust to mistralai SDK versions.
    """
    model = response.model
    usage = response.usage

    if model not in PRICING or usage is None:
        return 0.0

    if hasattr(usage, "input_tokens"):
        prompt_tokens = usage.input_tokens
        completion_tokens = usage.output_tokens
    elif hasattr(usage, "prompt_tokens"):
        prompt_tokens = usage.prompt_tokens
        completion_tokens = usage.completion_tokens
    else:
        return 0.0

    cost = (
        prompt_tokens / 1000 * PRICING[model]["input"] +
        completion_tokens / 1000 * PRICING[model]["output"]
    )
    return round(cost, 6)



# -----------------------------
# Plot confusion matrix
# -----------------------------

def plot_confusion_matrix(y_true, y_pred, labels, output_dir, dataset, model, run, shot):
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels)
    plt.title(f"Confusion Matrix\n{dataset} {model} Run {run}, {shot}-shot")
    plt.xlabel("Predicted")
    plt.ylabel("True")

    png_path = os.path.join(output_dir, f"{dataset}_run{run}_{shot}shot_{model}_confusion.png")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close()

    cm_df = pd.DataFrame(cm, index=[f"True_{l}" for l in labels],
                              columns=[f"Pred_{l}" for l in labels])
    csv_path = os.path.join(output_dir, f"{dataset}_run{run}_{shot}shot_{model}_confusion.csv")
    cm_df.to_csv(csv_path, index=True)


# -----------------------------
# Metrics
# -----------------------------

def compute_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=["Diabetic Retinopathy", "Normal"])

    if cm.shape == (2, 2):
        TP, FN = cm[0]
        FP, TN = cm[1]
    else:
        TP = FN = FP = TN = 0

    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0

    return acc, sensitivity, specificity


# -----------------------------
# Mistral inference functions
# -----------------------------

def zeroshot(client, model, query_prompt, system_prompt, query_image_path):
    encoded_query = encode_image_base64(query_image_path)

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": query_prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_query}"}}
            ]
        }
    ]

    for _ in range(5):
        try:
            response = client.chat.complete(
                model=model,
                messages=messages,
                temperature=0
            )

            content = response.choices[0].message.content
            cost = calculate_chat_cost(response)

            if content.strip():
                return content, cost
        except Exception as e:
            print(f"Mistral error: {e}")
            time.sleep(2)

    return "", 0.0


def fewshot(client, model, query_prompt, system_prompt, query_image_path, fewshot_examples):
    """
    Pixtral hard limit: max 8 images per request (including query image).
    We therefore cap few-shot examples to (7) images total.
    """

    MAX_IMAGES = 8

    # Cap few-shot examples to respect Pixtral limit
    max_fewshot = max(0, MAX_IMAGES - 1)  # reserve 1 for query image
    fewshot_examples = fewshot_examples[:max_fewshot]

    messages = [{"role": "system", "content": system_prompt}]

    for idx, (ex_path, ex_label) in enumerate(fewshot_examples, 1):
        encoded = encode_image_base64(ex_path)
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": f"Example {idx}: Diagnosis is {ex_label}."},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}}
            ]
        })

    encoded_query = encode_image_base64(query_image_path)
    messages.append({
        "role": "user",
        "content": [
            {"type": "text", "text": query_prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_query}"}}
        ]
    })

    for _ in range(5):
        try:
            response = client.chat.complete(
                model=model,
                messages=messages,
                temperature=0
            )

            content = response.choices[0].message.content
            cost = calculate_chat_cost(response)

            if content.strip():
                return content, cost
        except Exception as e:
            print(f"Mistral error: {e}")
            time.sleep(2)

    return "", 0.0


# -----------------------------
# CLI arguments
# -----------------------------

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--shot", type=int, required=True)
parser.add_argument("-r", "--run", type=int, required=True)
parser.add_argument("-m", "--model", required=True)
parser.add_argument("-d", "--dataset", required=True)
parser.add_argument("-k", "--key", type=int, default=0)
parser.add_argument("--fewshot-mode", choices=["single", "twopass"], default="single",
                    help="Few-shot strategy: single (default) or twopass (vote across two calls)")
args = parser.parse_args()

shot = args.shot
run = args.run
model = args.model
dataset = args.dataset
key_num = args.key
fewshot_mode = args.fewshot_mode

# -----------------------------
# Dataset paths (UNCHANGED)
# -----------------------------

if dataset == 'IDRiD516_orig':
    DR_dir = "/mnt/d/Naved/Data/IDRiD/idrid516_orig/dr_class"
    Normal_dir = "/mnt/d/Naved/Data/IDRiD/idrid516_orig/nm_class"
    output_dir = "/mnt/d/Naved/Outputs/idrid516_orig/"
else:
    raise ValueError("Dataset not configured")

os.makedirs(output_dir, exist_ok=True)

# -----------------------------
# Init client
# -----------------------------

client = Mistral(api_key=keychain[key_num])

classes = ["Diabetic Retinopathy", "Normal"]

# -----------------------------
# Prompts
# -----------------------------

system_prompt = (
    "You are a professional ophthalmologist. "
    "Classify the fundus image as Diabetic Retinopathy or Normal. "
    "Respond ONLY in valid JSON with keys 'thoughts' and 'answer'."
)

zeroshot_query_prompt = (
    "Analyze the fundus image carefully and classify it as either "
    "Diabetic Retinopathy or Normal."
)

fewshot_query_prompt = zeroshot_query_prompt

# -----------------------------
# Load images
# -----------------------------

def load_images(directory, label):
    return [(f, label) for f in os.listdir(directory)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))]

DR_images = load_images(DR_dir, classes[0])
NM_images = load_images(Normal_dir, classes[1])
all_images = DR_images + NM_images

# -----------------------------
# Main loop
# -----------------------------

results = {}
y_true, y_pred = [], []
unknown_count = 0
total_cost = 0

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

for idx, (image_name, label) in enumerate(all_images, 1):
    query_path = os.path.join(DR_dir if label == classes[0] else Normal_dir, image_name)

    available_mm = [img for img in DR_images if img[0] != image_name]
    available_nm = [img for img in NM_images if img[0] != image_name]

    fewshot_examples = []
    if shot > 0:
        fewshot_examples.extend(random.sample(available_mm, min(shot, len(available_mm))))
        fewshot_examples.extend(random.sample(available_nm, min(shot, len(available_nm))))

        fewshot_examples = [
            (os.path.join(DR_dir if lbl == classes[0] else Normal_dir, fname), lbl)
            for fname, lbl in fewshot_examples
        ]

    if shot == 0:
        resp, cost = zeroshot(client, model, zeroshot_query_prompt, system_prompt, query_path)
    else:
        if fewshot_mode == "single":
            resp, cost = fewshot(client, model, fewshot_query_prompt, system_prompt, query_path, fewshot_examples)
        else:
            # --- two-pass few-shot voting ---
            mid = len(fewshot_examples) // 2
            fs1 = fewshot_examples[:mid]
            fs2 = fewshot_examples[mid:]

            resp1, cost1 = fewshot(client, model, fewshot_query_prompt, system_prompt, query_path, fs1)
            resp2, cost2 = fewshot(client, model, fewshot_query_prompt, system_prompt, query_path, fs2)

            cost = cost1 + cost2

            # parse answers
            def extract_answer(r):
                try:
                    j = json.loads(clean_response(sanitize(r)))
                    return j.get("answer", "Unknown")
                except Exception:
                    return "Unknown"

            a1 = extract_answer(resp1)
            a2 = extract_answer(resp2)

            # majority vote
            if a1 == a2:
                resp = resp1
            else:
                resp = resp1  # tie-break: first pass


    total_cost += cost

    cleaned = clean_response(sanitize(resp))
    try:
        parsed = json.loads(cleaned)
    except Exception:
        parsed = {"thoughts": cleaned, "answer": "Unknown"}

    answer = parsed.get("answer", "Unknown")

    if "diabetic" in answer.lower():
        prediction = classes[0]
    elif "normal" in answer.lower():
        prediction = classes[1]
    else:
        prediction = "Unknown"
        unknown_count += 1

    y_true.append(label)
    y_pred.append(prediction)

    results[image_name] = parsed

    print(f"{idx}: {dataset}: {image_name} {shot}shot y_true:{label}, y_pred:{prediction} thoughts:{parsed.get('thoughts','')} Total cost:${total_cost}")

# -----------------------------
# Save results
# -----------------------------

json_path = os.path.join(
    output_dir,
    f"{dataset}_run{run}_{shot}shot_{model}_{timestamp}.json"
)

with open(json_path, "w") as f:
    json.dump(results, f, indent=2)

acc, sens, spec = compute_metrics(y_true, y_pred)
print(f"Accuracy:{acc:.2f} Sensitivity:{sens:.2f} Specificity:{spec:.2f}")
print(f"Unknown predictions: {unknown_count}")

plot_confusion_matrix(y_true, y_pred, classes, output_dir, dataset, model, run, shot)


"""

python3 idrid_mistral.py -s 0 -r 1 -m pixtral-large-2411 -d IDRiD516_orig -k 0 
python3 idrid_mistral.py -s 0 -r 2 -m pixtral-large-2411 -d IDRiD516_orig -k 1
python3 idrid_mistral.py -s 0 -r 3 -m pixtral-large-2411 -d IDRiD516_orig -k 2

python3 idrid_mistral.py -s 5 -r 1 -m pixtral-large-2411 -d IDRiD516_orig -k 0 
python3 idrid_mistral.py -s 5 -r 2 -m pixtral-large-2411 -d IDRiD516_orig -k 1
python3 idrid_mistral.py -s 5 -r 3 -m pixtral-large-2411 -d IDRiD516_orig -k 2


"""