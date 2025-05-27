from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import os


def load_text(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def extract_entities(text: str):
    model_name = "DeepPavlov/rubert-base-cased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name)

    ner = pipeline("ner", model=model, tokenizer=tokenizer, grouped_entities=True)
    results = ner(text)

    entities = [(r["word"], r["entity_group"]) for r in results]
    return entities


if __name__ == "__main__":
    file_path = "data/processed_texts/sample_medical_record_10_pages.txt"

    if not os.path.exists(file_path):
        print("⚠️ Файл не найден:", file_path)
        exit()

    print(f"📄 Обрабатывается файл: {file_path}")
    text = load_text(file_path)

    print("🔍 Извлекаются сущности...")
    entities = extract_entities(text[:1000])

    print("\n🧠 Найденные сущности:")
    for word, label in entities:
        print(f"{word} → {label}")
