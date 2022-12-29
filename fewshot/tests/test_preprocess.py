import pytest
from scripts.preprocess import replace_urls

def test_preprocess():
    in_text = '58,"Ah ... 🧠 https://t.co/lShlgWYGOX",1,Wood ... addressed ,0,1,0,0,0,0'
    expected = '58,"Ah ... 🧠 [URL]",1,Wood ... addressed ,0,1,0,0,0,0'

    assert replace_urls(in_text) == expected
