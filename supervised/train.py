import os

import evaluate
import numpy as np
import pandas as pd
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

os.environ["CUDA_VISIBLE_DEVICES"] = "5"
np.random.seed(100)

DEVICE = "cuda"
MODEL = "xlm-roberta-large"

isarcasm_data = pd.read_csv("data/isarcasm/preprocessed/train.csv")[:] #1800
isarcasm_data_ja = pd.read_csv("data/isarcasm/preprocessed/train_ja.csv")[:]
isarcasm_test_data = pd.read_csv("data/isarcasm/preprocessed/test.csv").sort_values(
    ["sarcastic"], ascending=False
)[:400]
isarcasm_test_data_ja = pd.read_csv(
    "data/isarcasm/preprocessed/test_ja.csv"
).sort_values(["sarcastic"], ascending=False)[:] #400
isarcasm_data = pd.concat([isarcasm_data, isarcasm_test_data])

spirs_data = pd.read_csv("data/spirs/preprocessed/all.csv")
spirs_data_ja = pd.read_csv("data/spirs/preprocessed/all_ja.csv")
spirs_data = pd.concat([spirs_data, spirs_data_ja])

chin_data = pd.read_csv("data/chinese/preprocessed/all.csv")
chin_data_ja = pd.read_csv("data/chinese/preprocessed/all_ja.csv")
chin_data = pd.concat([chin_data, chin_data_ja])

train_data = pd.concat([isarcasm_data, spirs_data, chin_data]).dropna()
train_data = train_data.sample(frac=1).reset_index()

eval_data = train_data[-1500:]
eval_data.to_csv("data/evaluation/all.csv")
train_data = train_data[:-1500]

print((train_data['sarcastic'] == 1).sum() / len(train_data['sarcastic']))
print((eval_data['sarcastic'] == 1).sum() / len(eval_data['sarcastic']))
# raise

tokenizer = AutoTokenizer.from_pretrained(MODEL, cache_dir="./cache")

recall = evaluate.load("recall")
prec = evaluate.load("precision")
f_score = evaluate.load("f1")
accuracy = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    print(np.mean(predictions))
    predictions = predictions >= 0.5
    r = recall.compute(predictions=predictions, references=labels)["recall"]
    p = prec.compute(predictions=predictions, references=labels)["precision"]
    f = f_score.compute(predictions=predictions, references=labels)["f1"]
    a = accuracy.compute(predictions=predictions, references=labels)["accuracy"]
    return {"precision": p, "recall": r, "f1": f, "accuracy" : a}


class SarcasmDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.text = data["text"].tolist()
        self.encodings = tokenizer(
            self.text, padding=True, truncation=True, max_length=500
        )
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

training_args = TrainingArguments(
    output_dir="./checkpoints",
    num_train_epochs=25,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=2500,
    learning_rate=1e-5,
    weight_decay=0.0001,
    logging_dir="./logs",
    logging_steps=100,
    evaluation_strategy="epoch",
    fp16=True,
    save_steps=5000,
    seed= 100,
    data_seed= 100,
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
