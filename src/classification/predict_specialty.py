import argparse
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F

parser = argparse.ArgumentParser()
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument("--text", type=str, help="Медицинский текст")
group.add_argument("--file", type=str, help="Путь к .txt файлу (анализ по строкам)")
args = parser.parse_args()

model_path = "models/classifier"
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)
model.eval()

id2label = model.config.id2label
if isinstance(id2label, dict):
    id2label = {int(k): v for k, v in id2label.items()}

def classify(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1)
        sorted_indices = torch.argsort(probs[0], descending=True)
        return [(id2label[i.item()], probs[0][i].item()) for i in sorted_indices[:3]]

if args.text:
    print(f"\n📄 Текст: {args.text}")
    for label, score in classify(args.text):
        print(f"→ {label:<15} {score * 100:.1f}%")

if args.file:
    with open(args.file, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if len(line.strip()) > 30]
    for i, line in enumerate(lines, 1):
        print(f"\n🧾 Фрагмент {i}: {line}")
        for label, score in classify(line):
            print(f"→ {label:<15} {score * 100:.1f}%")
