#!/bin/bash


GPU=1

MT_LIST=(bert bert bert bert bert)
MODEL_PATH_LIST=(pretrained_models/bert_voken pretrained_models/voken_base pretrained_models/vidlankd pretrained_models/bert_wiki_coco pretrained_models/bert_wiki_howto100m)
NAME_LIST=(bert bert-i bert-v voken vidlankd)

for SEED in 0 2021 27; do
    for BATCH in 256; do
        for EPOCH in 5; do
            for(( i=0;i<${#MT_LIST[@]};i++)); do
                TYPE=${MT_LIST[$i]}
                MODEL_PATH=${MODEL_PATH_LIST[$i]}
                NAME=${NAME_LIST[$i]}
            
                # echo $MT $MODEL_PATH $NAME $PRED
                python commonsense_analysis/main_kb_pt.py --model_type $TYPE --model_name_or_path $MODEL_PATH --save_dir commonsense_analysis/res/atomic2020/$NAME/t\_$BATCH\_$EPOCH\_$SEED --pred t --epoch $EPOCH --batch_size $BATCH --gpu_mode --gpu_index $GPU --fp16 --seed $SEED
            done
        done
    done
done