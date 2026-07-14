"""
Task 1: News Topic Classifier Using BERT
Fine-tune BERT on AG News Dataset. Optimized for CPU.
"""

import torch, numpy as np, json, os
from datasets import load_dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, f1_score, classification_report

LABEL_MAP = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}
OUTPUT_DIR = "./news_topic_model"

print("=" * 60)
print("Task 1: News Topic Classifier Using BERT")
print("=" * 60)

dataset = load_dataset("fancyzhx/ag_news")
train_ds = dataset["train"].shuffle(seed=42).select(range(3000))
test_ds = dataset["test"].shuffle(seed=42).select(range(1000))
print(f"Train: {len(train_ds)}, Test: {len(test_ds)}")

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=32)

train_ds = train_ds.map(tokenize, batched=True, batch_size=1000)
test_ds = test_ds.map(tokenize, batched=True, batch_size=1000)
train_ds.set_format("torch", columns=["input_ids", "attention_mask", "label"])
test_ds.set_format("torch", columns=["input_ids", "attention_mask", "label"])

model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=4)
model.to("cpu")

def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=-1)
    return {"accuracy": accuracy_score(labels, preds), "f1": f1_score(labels, preds, average="weighted")}

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR, num_train_epochs=2, per_device_train_batch_size=32,
    per_device_eval_batch_size=64, warmup_steps=50, weight_decay=0.01,
    logging_steps=50, eval_strategy="epoch", save_strategy="epoch",
    load_best_model_at_end=True, report_to="none", use_cpu=True,
)

trainer = Trainer(model=model, args=training_args, train_dataset=train_ds, eval_dataset=test_ds, compute_metrics=compute_metrics)
trainer.train()

results = trainer.evaluate(test_ds)
print(f"\nTest Accuracy: {results['eval_accuracy']:.4f}")
print(f"Test F1-Score: {results['eval_f1']:.4f}")

preds_output = trainer.predict(test_ds)
y_pred = np.argmax(preds_output.predictions, axis=-1)
print("\nClassification Report:")
print(classification_report(preds_output.label_ids, y_pred, target_names=list(LABEL_MAP.values())))

model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
with open(os.path.join(OUTPUT_DIR, "metrics.json"), "w") as f:
    json.dump({"accuracy": results["eval_accuracy"], "f1": results["eval_f1"]}, f, indent=2)

print(f"\nModel saved to {OUTPUT_DIR}")
print("=" * 60)
print("Task 1 COMPLETE")
print("=" * 60)
