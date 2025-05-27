import pandas as pd
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import DataCollatorWithPadding
import torch

# === 1. Загрузка нового датасета ===
df = pd.read_csv("data/classification/rumed_classification_dataset.csv")  # путь к новому файлу
df = df[df["text"].notna() & df["specialty"].notna()]  # фильтрация

# Преобразуем специальности в числовые метки
label_encoder = LabelEncoder()
df["label"] = label_encoder.fit_transform(df["specialty"])

# Маппинги
label2id = {label: idx for idx, label in enumerate(label_encoder.classes_)}
id2label = {idx: label for label, idx in label2id.items()}

# HuggingFace Dataset
dataset = Dataset.from_pandas(df[["text", "label"]].sample(frac=1).reset_index(drop=True))
dataset = dataset.train_test_split(test_size=0.2)

# === 2. Токенизация ===
model_name = "DeepPavlov/rubert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def preprocess(example):
    return tokenizer(example["text"], truncation=True, max_length=512)

encoded_dataset = dataset.map(preprocess, batched=True)

# === 3. Модель ===
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=len(label2id),
    id2label=id2label,
    label2id=label2id
)

# === 4. Аргументы обучения ===
training_args = TrainingArguments(
    output_dir="./models/classifier",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    logging_dir="./logs",
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    save_total_limit=1,
    dataloader_num_workers=0  # важно для macOS
)

# === 5. Trainer ===
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.argmax(torch.tensor(logits), dim=-1)
    acc = (predictions == torch.tensor(labels)).float().mean()
    return {"accuracy": acc.item()}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# === 6. Запуск ===
if __name__ == "__main__":
    trainer.train()
    trainer.save_model("./models/classifier")
    tokenizer.save_pretrained("./models/classifier")
