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

DEVICE = "cuda"
MODEL = "results/checkpoint-28000"

tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-large", cache_dir="./cache")
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL, num_labels=1, cache_dir="./cache"
).to(DEVICE)

pred_data = pd.read_csv("data/prediction/all.csv", names=['text'], delimiter='▞')[:25]


from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer

pipe = pipeline('text-classification', model=MODEL, tokenizer='xlm-roberta-large')

preds = pipe(pred_data['text'].tolist(), function_to_apply=None)
preds = list(map(lambda x: x['score'],preds))

print(preds)