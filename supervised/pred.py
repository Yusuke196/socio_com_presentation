import os

import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

os.environ["CUDA_VISIBLE_DEVICES"] = "5"

DEVICE = "cuda"
MODEL = "results/checkpoint-35000"

tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-large", cache_dir="./cache")
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL, num_labels=1, cache_dir="./cache"
).to(DEVICE)

pred_data = pd.read_csv("data/prediction/all.csv", names=["text"], delimiter="▞")[:25]

pipe = pipeline("text-classification", model=MODEL, tokenizer="xlm-roberta-large")

preds = pipe(pred_data["text"].tolist(), function_to_apply=None)
preds = list(map(lambda x: 0 if x["score"] <= 0.5 else 1, preds))

pred_data["sarcastic"] = preds

pred_data.to_csv("supervised_preds.csv")
