from io import StringIO
import pandas as pd
import re


def main():
    for csv in ['train.csv']:
        with open(f'data/raw/{csv}') as in_file:
            text = replace_urls(in_file.read())
            df = pd.read_csv(StringIO(text), on_bad_lines='skip', index_col=0)
            # Match column names with test.csv
            df = df.iloc[:, :2].rename(columns={'tweet': 'text'})
            df.to_csv(f'data/preprocessed/{csv}', index=False)


def replace_urls(text):
    url_pattern = r'https?://\S+'
    return re.sub(url_pattern, '[URL]', text)


if __name__ == '__main__':
    main()
