import pandas as pd
from transformers import (
    MT5ForConditionalGeneration,
    T5Tokenizer,
    Text2TextGenerationPipeline,
)

DATA = "data/chinese/preprocessed/all.csv"

path = "K024/mt5-zh-ja-en-trimmed"
pipe = Text2TextGenerationPipeline(
    model=MT5ForConditionalGeneration.from_pretrained(path, cache_dir="./cache"),
    tokenizer=T5Tokenizer.from_pretrained(path, cache_dir="./cache"),
    device="cuda:7",
)

df = pd.read_csv(DATA).dropna()
df["text"] = "zh2ja: " + df["text"]
df["text"] = list(
    map(
        lambda x: x["generated_text"],
        pipe(df["text"].tolist(), max_length=300, num_beams=4),
    )
)
df.to_csv("all_ja.csv", index=False)
