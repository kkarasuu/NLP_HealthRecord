import argparse
import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F

parser = argparse.ArgumentParser()
parser.add_argument("--file", type=str, required=True, help="Путь к .txt файлу")
parser.add_argument("--specialty", type=str, required=True, help="Целевая специальность")
parser.add_argument("--threshold", type=float, default=0.3, help="Порог вероятности")
parser.add_argument("--context", type=int, default=1, help="Сколько фрагментов после включать")
args = parser.parse_args()

model_path = "models/classifier"
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)
model.eval()

id2label = model.config.id2label
if isinstance(id2label, dict):
    id2label = {int(k): v for k, v in id2label.items()}
label2id = {v: k for k, v in id2label.items()}

if args.specialty not in label2id:
    raise ValueError(f"❌ Специальность '{args.specialty}' не найдена. Доступные: {list(label2id.keys())}")

target_id = label2id[args.specialty]

with open(args.file, "r", encoding="utf-8") as f:
    fragments = [line.strip() for line in f if len(line.strip()) > 30]

selected_indices = set()

for i, frag in enumerate(fragments):
    inputs = tokenizer(frag, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1)
        score = probs[0][target_id].item()

    if score >= args.threshold:
        for j in range(i, min(i + 1 + args.context, len(fragments))):
            selected_indices.add(j)

basename = os.path.basename(args.file).replace(".txt", "")
output_path = f"data/filtered/{basename}__filtered__{args.specialty}.txt"
os.makedirs(os.path.dirname(output_path), exist_ok=True)

with open(output_path, "w", encoding="utf-8") as f:
    for idx in sorted(selected_indices):
        f.write(fragments[idx] + "\n\n")

# print(f"Сохранено {len(selected_indices)} фрагментов с контекстом → {output_path}")
with open(output_path, "r", encoding="utf-8") as f:
    print(f.read())
