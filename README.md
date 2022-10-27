# Vision Analysis
Codes for the paper **What You See Helps You Read: Understanding How Vision Enhances Language Semantics, Relations and Commonsense**
## Pretrain BERT Baseline Models for Compare
Check [here](https://github.com/VickiCui/vision_analysis/tree/main/pretrain_bert) for scripts to run model pretraining.

## Word Embedding Analysis
The code for probing semantic shift, show word neighbors and association analysis is at `embedding_analysis.ipynb`.
download all datasets for word embedding analysis, and put them into `./data` dir.

script for test word analogy:
```
bash embedding_analysis/scripts/run_analogy.sh
```

script for test semantic-classes classification:
```
bash embedding_analysis/scripts/run_sclass.sh
```

script for test WiC word sense disambiguation task:
```
bash embedding_analysis/scripts/run_superglue.sh
```

## Relation Analysis
### Semantic Relation
script for test SemEval2010 task 8 relation classification task:
```
bash relation_analysis/scripts/run_relclass.sh
```

### Coherence Relation
The script for test SentEval is:
```
bash relation_analysis/scripts/run_senteval.sh
```

## Commonsense Analysis
### Sentence Representation Analysis
Speech, Language and the Brain (CSLB) property norms data should be manually download from https://cslb.psychol.cam.ac.uk/propnorms
After downloading, you should unzip the data to `./data/` and run this script to pre-process the data:
```
bash commonsense_analysis/get_sentence_data.sh
```

The script for test CSLB is:
```
python commonsense_analysis/run_cslb.sh
```

### Knowledge Base Analysis
We use "Masked Language Model Scoring" to probe to what extent does a model master some commonsense knowledge.
The script of running ConceptNet/Atomic analysis for each model is:
```
python commonsense_analysis/run_atomic2020.sh
```