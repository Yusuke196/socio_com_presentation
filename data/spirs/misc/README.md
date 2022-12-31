# SPIRS Sarcasm Dataset

SPIRS is a unique dataset of 15,000 sarcastic tweets. SPIRS was collected using **reactive supervision**, a new data capturing method. 
Reactive supervision allows the collection of both _intended sarcasm_ and _perceived sarcasm_ texts. 

**SPIRS** stands for **S**arcasm, **P**erceived and **I**ntended, by **R**eactive **S**upervision :)

To find out more about SPIRS and reactive supervision, check out the [reactive supervision paper](https://arxiv.org/abs/2009.13080), or read the [Medium article](https://towardsdatascience.com/the-magic-of-reactive-supervision-3fc83cdb1ca4). Or watch this short, 7-minute [YouTube video about reactive supervision](https://www.youtube.com/watch?v=Wx6S-KdZ1nM).

Use this repository to download SPIRS. The repository includes the following data files:

  * `SPIRS-sarcastic-ids.csv` the sarcastic tweet IDs (15,000 "positive" samples)
  * `SPIRS-non-sarcastic-ids.csv` the non-sarcastic tweet IDs (15,000 "negative" samples)
  
Additional fields for each sarcastic tweet include the sarcasm perspective (intended/perceived), author sequence, and contextual tweet IDs (cue, oblivious, and eliciting tweets).
More information is available in the reactive supervision paper.

To comply with Twitter's privacy policy, the dataset files include only the tweet IDs. To fetch the tweet texts, follow these steps:

  * Install the latest version of Tweepy:
  
    `pip3 install tweepy`
  * Rename our `credentials-example.py` to `credentials.py`
  * Add your Twitter API credentials by editing `credentials.py`
  * Run the script:
  
    `python3 fetch-tweets.py`

The script will fetch the texts and create two new files, one for sarcastic and the other for non-sarcastic tweets:

  * `SPIRS-sarcastic.csv`
  * `SPIRS-non-sarcastic.csv`

## Citation

Kindly cite the paper using the following BibTex entry:

```
@inproceedings{shmueli-etal-2020-reactive,
    title = "{R}eactive {S}upervision: {A} {N}ew {M}ethod for {C}ollecting {S}arcasm {D}ata",
    author = "Shmueli, Boaz  and
      Ku, Lun-Wei  and
      Ray, Soumya",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)",
    month = nov,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.emnlp-main.201",
    doi = "10.18653/v1/2020.emnlp-main.201",
    pages = "2553--2559",
    abstract = "Sarcasm detection is an important task in affective computing, requiring large amounts of labeled data. We introduce reactive supervision, a novel data collection method that utilizes the dynamics of online conversations to overcome the limitations of existing data collection techniques. We use the new method to create and release a first-of-its-kind large dataset of tweets with sarcasm perspective labels and new contextual features. The dataset is expected to advance sarcasm detection research. Our method can be adapted to other affective computing domains, thus opening up new research opportunities.",
}
```

