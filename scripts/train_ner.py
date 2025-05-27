from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments
from datasets import load_dataset, Dataset
from sklearn.metrics import classification_report
import json
import numpy as np
import torch

if not hasattr(torch, "get_default_device"):
    def get_default_device():
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.get_default_device = get_default_device

model_name = "emilyalsentzer/Bio_ClinicalBERT"
tokenizer = AutoTokenizer.from_pretrained(model_name)

with open("ner_train_large.json") as f:
    data = json.load(f)

dataset = Dataset.from_list(data)
label_names = ["O", "Diagnosis", "Symptom", "Treatment", "Procedure", "TestResult"]
num_labels = len(label_names)

def tokenize_and_align_labels(example):
    tokenized_inputs = tokenizer(example["tokens"], truncation=True, padding='max_length', max_length=128, is_split_into_words=True)
    word_ids = tokenized_inputs.word_ids()
    labels = []
    for word_id in word_ids:
        if word_id is None:
            labels.append(-100)
        else:
            labels.append(example["ner_tags"][word_id])
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

encoded_dataset = dataset.map(tokenize_and_align_labels)
dataset_split = encoded_dataset.train_test_split(test_size=0.1)

model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=num_labels)
args = TrainingArguments(output_dir="category_classifier", per_device_train_batch_size=16, per_device_eval_batch_size=16, num_train_epochs=4, report_to=["none"])

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset_split['train'],
    eval_dataset=dataset_split['test']
)

trainer.train()

model.save_pretrained("category_classifier")
tokenizer.save_pretrained("category_classifier")

predictions = trainer.predict(dataset_split['test'])
y_true = predictions.label_ids
y_pred = np.argmax(predictions.predictions, axis=-1)

true_entities, pred_entities = [], []
for true, pred in zip(y_true, y_pred):
    for t, p in zip(true, pred):
        if t != -100:
            true_entities.append(t)
            pred_entities.append(p)

print(classification_report(true_entities, pred_entities, target_names=label_names))
