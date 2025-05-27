import torch
import json
import fitz  # PyMuPDF
import argparse
import os
import nltk
import nltk.data
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForTokenClassification, pipeline
from nltk.tokenize.punkt import PunktSentenceTokenizer
import pickle

nltk.download('punkt_tab')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

# === Аргументы командной строки ===
parser = argparse.ArgumentParser(description="Medical text inference pipeline")
parser.add_argument("--pdf", type=str, required=True, help="Path to input PDF file")
parser.add_argument("--specialty", type=str, required=True, help="Target medical specialty")
parser.add_argument("--output", type=str, default="output.json", help="Path to save the result JSON")
args = parser.parse_args()

# === Загрузка PDF ===
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    return "\n".join(page.get_text() for page in doc)

# === Пути к моделям
spec_model_path = "models/specialty_filter"
cat_model_path = "models/category_classifier"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Загрузка моделей
spec_model = AutoModelForSequenceClassification.from_pretrained(spec_model_path, local_files_only=True).to(device)
spec_tokenizer = AutoTokenizer.from_pretrained(spec_model_path, local_files_only=True)
cat_model = AutoModelForTokenClassification.from_pretrained(cat_model_path, local_files_only=True).to(device)
cat_tokenizer = AutoTokenizer.from_pretrained(cat_model_path, local_files_only=True)

ner = pipeline("ner", model=cat_model, tokenizer=cat_tokenizer, aggregation_strategy="simple", device=0 if torch.cuda.is_available() else -1)

# === Загрузка Punkt Sentence Tokenizer
try:
    tokenizer_path = "/usr/share/nltk_data/tokenizers/punkt/english.pickle"
    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)
except Exception as e:
    print("[ERROR] Failed to load Punkt tokenizer:", e)
    tokenizer = PunktSentenceTokenizer()

# === Фильтрация по специальности
def is_relevant(sentence, target_class, label2id):
    inputs = spec_tokenizer(sentence, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        logits = spec_model(**inputs).logits
        predicted = torch.argmax(logits, dim=1).item()
    return predicted == label2id[target_class]

# === Основной пайплайн
def run_pipeline(pdf_path, target_specialty):
    print("=== 🔍 Запуск пайплайна ===")
    text = extract_text_from_pdf(pdf_path)
    print(f"[DEBUG] Первый абзац текста:\n{text[:500]}")

    sentences = tokenizer.tokenize(text)
    print(f"[DEBUG] Кол-во предложений: {len(sentences)}")

    with open(os.path.join(spec_model_path, "config.json")) as f:
        config = json.load(f)
        label2id = config["label2id"]
    print(f"[DEBUG] Метки: {list(label2id.keys())}")

    relevant = [s for s in sentences if is_relevant(s, target_specialty, label2id)]
    print(f"[DEBUG] Релевантных предложений: {len(relevant)}")

    result = {
        "specialty": target_specialty,
        "diagnoses": [],
        "complaints": [],
        "procedures": [],
        "treatments": [],
        "test_results": []
    }

    label_map = {
        "LABEL_1": "Diagnosis",
        "LABEL_2": "Symptom",
        "LABEL_3": "Treatment",
        "LABEL_4": "Procedure",
        "LABEL_5": "TestResult"
    }

    mapping = {
        "Diagnosis": "diagnoses",
        "Symptom": "complaints",
        "Treatment": "treatments",
        "Procedure": "procedures",
        "TestResult": "test_results"
    }

    bad_phrases = {
        "section", "sections", "conclusion", "personal", "history", "name", "date", "address",
        "signature", "bp", "bpm", "doctor", "physician", "institution"
    }

    for sentence in relevant:
        print(f"[DEBUG] → NER на: {sentence}")
        entities = ner(sentence)
        print(f"[DEBUG] ← Сущности: {entities}")

        for ent in entities:
            group = ent["entity_group"]
            label = label_map.get(group)
            if not label:
                continue

            raw = ent["word"].strip()
            if len(raw) < 4 or any(bad in raw.lower() for bad in bad_phrases):
                continue

            tokens = [t.strip() for part in raw.split(" and ") for t in part.replace(";", ",").split(",")]
            for token in tokens:
                if len(token) < 4 or any(bad in token.lower() for bad in bad_phrases):
                    continue
                if token not in result[mapping[label]]:
                    result[mapping[label]].append(token)

    return result

# === Точка входа
if __name__ == "__main__":
    output = run_pipeline(args.pdf, args.specialty)

    print("\n=== 🧩 Финальный результат:")
    print(json.dumps(output, indent=2, ensure_ascii=False))

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Результат сохранён в {args.output}")
