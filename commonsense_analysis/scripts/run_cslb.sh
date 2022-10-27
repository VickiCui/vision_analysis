N=1
MT_LIST=(bert bert bert bert bert)
MODEL_PATH_LIST=(pretrained_models/bert_voken pretrained_models/voken_base pretrained_models/vidlankd pretrained_models/bert_wiki_coco pretrained_models/bert_wiki_howto100m)
NAME_LIST=(bert bert-i bert-v voken vidlankd)

for(( i=0;i<${#MT_LIST[@]};i++)); do
    TYPE=${MT_LIST[$i]}
    MODEL_PATH=${MODEL_PATH_LIST[$i]}
    NAME=${NAME_LIST[$i]}
    python commonsense_analysis/sentence.py \
        --model_name_or_path $MODEL_PATH \
        --model_type $TYPE \
        --norm \
        --save_path commonsense_analysis/res/sentence_representation/${NAME}.json \
        --gpu_mode \
        --gpu_index $N
done