import pandas as pd
from transformers import pipeline

DATA = "data/isarcasm/preprocessed/train.csv"
fugu_translator = pipeline("translation", model="staka/fugumt-en-ja", device="cuda:7")

df = pd.read_csv(DATA).dropna()
df["text"] = list(
    map(lambda x: x["translation_text"], fugu_translator(df["text"].tolist()))
)

df.to_csv("train_ja.csv", index=False)
