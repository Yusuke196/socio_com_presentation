import pandas as pd
import pytest
from scripts.preprocess import replace_urls

def test_replace_urls():
    in_text = '58,"Ah ... 🧠 https://t.co/lShlgWYGOX",1,Wood ... addressed ,0,1,0,0,0,0'
    expected = '58,"Ah ... 🧠 [URL]",1,Wood ... addressed ,0,1,0,0,0,0'

    assert replace_urls(in_text) == expected

def test_preprocessed_csv():
    df = pd.read_csv('data/preprocessed/train.csv')
    # If this goes without an error, the csv can be assumed to be well-formed
    df.astype({'sarcastic':'int'})

    assert True
