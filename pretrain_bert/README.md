# pretrain BERT_base with text/text+image/text+video corpus
The BERT model parameters are initialized from scratch and pretrained on the corresponding data.
Wiki data can be download with:
```
mkdir -p data/wiki
wget  https://nlp.cs.unc.edu/data/vokenization/wiki-cased/en.test.raw.bert-base-uncased.hdf5 -P data/wiki
wget  https://nlp.cs.unc.edu/data/vokenization/wiki-cased/en.train.raw.bert-base-uncased.hdf5 -P data/wiki
wget  https://nlp.cs.unc.edu/data/vokenization/wiki-cased/en.valid.raw.bert-base-uncased.hdf5 -P data/wiki

```
Wiki+Coco and Wiki+HowTo100M data we used can be found at ![here](to be add). Please download unzip all data and put them into `data/wiki_coco` and `data/wiki_howto100m` respectively.

## English Wikipedia
```
bash scripts/base_wiki.bash 0,1 bert_wiki
```

## English Wikipedia + Image Captions
```
bash scripts/base_wiki_coco.bash 0,1 bert_wiki_coco
```


## English Wikipedia + Video Captions
```
bash scripts/base_wiki_howto100m.bash 0,1 bert_wiki_howto100m
```
Please note that since the howto100m dataset is collected through speech recognition, there will be issues such as segment repetition, no punctuation, etc. We performed the necessary preprocessing on the original howto100m dataset, the code can be found at `tools/add_punct.py`