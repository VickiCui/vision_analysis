
N=1
MT_LIST=(bert bert bert bert bert)
MODEL_PATH_LIST=(pretrained_models/bert_voken pretrained_models/voken_base pretrained_models/vidlankd pretrained_models/bert_wiki_coco pretrained_models/bert_wiki_howto100m)
NAME_LIST=(bert bert-i bert-v voken vidlankd)

for(( i=0;i<${#MT_LIST[@]};i++)); do
    TYPE=${MT_LIST[$i]}
    MODEL_PATH=${MODEL_PATH_LIST[$i]}
    NAME=${NAME_LIST[$i]}

    python linguistic_analysis/analogy.py --model_name_or_path $MODEL_PATH --save_path linguistic_analysis/res/bats_analogy/$NAME.json --data_type bats --data_path data/BATS_3.0 --gpu_mode --gpu_index $N

    python linguistic_analysis/analogy.py --model_name_or_path $MODEL_PATH --save_path linguistic_analysis/res/google_analogy/$NAME.json --data_type google --data_path data/google-analogies.csv --gpu_mode --gpu_index $N

done



