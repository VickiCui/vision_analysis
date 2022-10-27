GPU=1

MT_LIST=(bert bert bert bert bert)
MODEL_PATH_LIST=(pretrained_models/bert_voken pretrained_models/voken_base pretrained_models/vidlankd pretrained_models/bert_wiki_coco pretrained_models/bert_wiki_howto100m)
NAME_LIST=(bert bert-i bert-v voken vidlankd)

for(( i=0;i<${#MT_LIST[@]};i++)); do
    MT=${MT_LIST[$i]}
    MODEL_PATH=${MODEL_PATH_LIST[$i]}
    NAME=${NAME_LIST[$i]}
    for LAYER in {1..12..1}; do
        for TYPE in avg cls; do
            python others_analysis/sent_eval.py --model_type $MT --model_name_or_path $MODEL_PATH --save_path linguistic_analysis/res/sent_eval/$NAME\_$TYPE\_layer$LAYER --type_sent $TYPE --layer $LAYER --gpu_mode --gpu_index $GPU
        done
    done
done