from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset, Dataset
from sklearn.metrics import classification_report
import json
import os
import torch

if not hasattr(torch, "get_default_device"):
    def get_default_device():
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.get_default_device = get_default_device

model_name = "emilyalsentzer/Bio_ClinicalBERT"
tokenizer = AutoTokenizer.from_pretrained(model_name)

with open("specialty_train_large.json") as f:
    data = json.load(f)

data = [x for x in data if isinstance(x["text"], str) and isinstance(x["label"], str)]

dataset = Dataset.from_list(data)
label2id = {label: i for i, label in enumerate(sorted(set(d["label"] for d in data)))}
id2label = {v: k for k, v in label2id.items()}

dataset = dataset.map(lambda x: {"label": label2id[x["label"]]})

def tokenize(example):
    return tokenizer(example['text'], truncation=True, padding='max_length', max_length=128)

dataset = dataset.map(tokenize)
dataset = dataset.train_test_split(test_size=0.1)

model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(label2id), id2label=id2label, label2id=label2id)
args = TrainingArguments(output_dir="../models/specialty_filter", per_device_train_batch_size=16, per_device_eval_batch_size=16, num_train_epochs=4, report_to=["none"])

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset['train'],
    eval_dataset=dataset['test']
)

trainer.train()

model.save_pretrained("specialty_filter")
tokenizer.save_pretrained("specialty_filter")

predictions = trainer.predict(dataset['test'])
y_true = predictions.label_ids
y_pred = predictions.predictions.argmax(-1)
print(classification_report(y_true, y_pred, target_names=[id2label[i] for i in range(len(id2label))]))
