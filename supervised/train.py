import os

import pandas as pd
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
import evaluate

recall = evaluate.load("recall")
prec = evaluate.load("precision")
import numpy as np


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    print(np.mean(predictions))
    predictions = (predictions >= 0.5)
    r = recall.compute(predictions=predictions, references=labels)["recall"]
    p = prec.compute(predictions=predictions, references=labels)["precision"]
    return {"precision": p, "recall": r}

os.environ["CUDA_VISIBLE_DEVICES"] = "6"

DEVICE = "cuda"
MODEL = "xlm-roberta-large"

isarcasm_data = pd.read_csv("data/isarcasm/preprocessed/train.csv").fillna(" ")[:1800]
isarcasm_test_data = pd.read_csv("data/isarcasm/preprocessed/test.csv").sort_values(['sarcastic'], ascending=False)[:400]
# print(isarcasm_test_data.head())
# raise
isarcasm_data = pd.concat([isarcasm_data, isarcasm_test_data])
# isarcasm_data = isarcasm_data.sample(frac=1).reset_index()
spirs_data = pd.read_csv("data/spirs/preprocessed/all.csv")
train_data = pd.concat([isarcasm_data, spirs_data])
train_data = train_data.sample(frac=1).reset_index()

eval_data = train_data[-400:]
train_data = train_data[:-400]


tokenizer = AutoTokenizer.from_pretrained(MODEL, cache_dir="./cache")


class SarcasmDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.text = data["text"].tolist()
        self.encodings = tokenizer(self.text, padding=True)
        self.labels = data["sarcastic"].astype(float).tolist()

    def __getitem__(self, idx):
        item = {
            key: torch.tensor(val[idx], device="cpu")
            for key, val in self.encodings.items()
        }
        item["labels"] = torch.tensor(self.labels[idx], device="cpu")
        return item

    def __len__(self):
        return len(self.labels)


train_dataset = SarcasmDataset(train_data)
val_dataset = SarcasmDataset(eval_data)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL, num_labels=1, cache_dir="./cache"
).to(DEVICE)

# from torch import nn

# class CustomTrainer(Trainer):
#     def compute_loss(self, model, inputs, return_outputs=False):
#         labels = inputs.get("labels")
#         # forward pass
#         outputs = model(**inputs)
#         logits = outputs.get('logits')
#         # compute custom loss
#         loss_fct = nn.CrossEntropyLoss(weight=torch.tensor([0.2, 0.3]))
#         loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
#         return (loss, outputs) if return_outputs else loss

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=20,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    warmup_steps=50,
    learning_rate=5e-6,
    # weight_decay=0.001,
    logging_dir="./logs",
    logging_steps=100,
    evaluation_strategy="epoch",
    fp16=True,
    # deepspeed="supervised/ds-config.json"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()
