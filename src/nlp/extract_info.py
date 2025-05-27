import argparse
import json
import os
import re

parser = argparse.ArgumentParser()
parser.add_argument("--file", type=str, required=True, help="Путь к .txt файлу (фильтрованный)")
parser.add_argument("--out", type=str, default=None, help="Путь для сохранения JSON")
args = parser.parse_args()

with open(args.file, "r", encoding="utf-8") as f:
    text = f.read()

# === Шаблоны для поиска ===
symptom_keywords = ["жалуется", "беспокоит", "наблюдаются", "отмечает", "предъявляет"]
diagnosis_keywords = ["диагноз", "поставлен", "выставлен", "подозрение на"]
prescription_keywords = ["назначен", "назначена", "рекомендован", "прописан", "выписан"]
temperature_pattern = r"(\d{2}\.?\d*)\s*°?C"

# === Результирующая структура ===
info = {
    "Симптомы": [],
    "Температура": None,
    "Диагноз": None,
    "Назначения": []
}

# === Обработка ===
for line in text.split("\n"):
    lower = line.lower()

    if any(kw in lower for kw in symptom_keywords):
        info["Симптомы"].append(line.strip())

    if any(kw in lower for kw in diagnosis_keywords) and info["Диагноз"] is None:
        info["Диагноз"] = line.strip()

    if any(kw in lower for kw in prescription_keywords):
        info["Назначения"].append(line.strip())

    if info["Температура"] is None:
        match = re.search(temperature_pattern, line)
        if match:
            info["Температура"] = match.group(1) + "°C"

# Удалим пустые поля
if not info["Симптомы"]:
    info.pop("Симптомы")
if not info["Назначения"]:
    info.pop("Назначения")
if not info["Диагноз"]:
    info.pop("Диагноз")
if not info["Температура"]:
    info.pop("Температура")

# === Сохранение ===
out_path = args.out or args.file.replace(".txt", ".json")
os.makedirs(os.path.dirname(out_path), exist_ok=True)

with open(out_path, "w", encoding="utf-8") as f:
    json.dump(info, f, ensure_ascii=False, indent=2)

print(f"✅ Извлечённая структура сохранена: {out_path}")
