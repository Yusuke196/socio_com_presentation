import pandas as pd


def main():
    df = pd.read_table(
        "data/prediction/fewshot_all.tsv", names=["tweet", "pred"]
    )
    df = df[~df["tweet"].str.contains("[USER]")]
    df = df.query("pred == 1")


if __name__ == "__main__":
    main()
