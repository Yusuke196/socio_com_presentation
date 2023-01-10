import os

import pandas as pd
from transformers import pipeline

# os.environ["CUDA_VISIBLE_DEVICES"] = "5"

DEVICE = 6
MODEL = "checkpoints/checkpoint-50000"

pred_data = pd.read_csv("data/prediction/all.csv", names=["text"], delimiter="▞")[:]

pipe = pipeline("text-classification", model=MODEL, tokenizer="xlm-roberta-large", device=DEVICE)

preds = pipe(pred_data["text"].tolist(), function_to_apply=None)
preds = list(map(lambda x: x["score"], preds))

pred_data["sarcastic"] = preds

pred_data.to_csv("supervised_preds.csv")
