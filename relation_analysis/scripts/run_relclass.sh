N=1
MT_LIST=(bert bert bert bert bert)
MODEL_PATH_LIST=(pretrained_models/bert_voken pretrained_models/voken_base pretrained_models/vidlankd pretrained_models/bert_wiki_coco pretrained_models/bert_wiki_howto100m)
NAME_LIST=(bert bert-i bert-v voken vidlankd)

for(( i=0;i<${#MT_LIST[@]};i++)); do
    TYPE=${MT_LIST[$i]}
    MODEL_PATH=${MODEL_PATH_LIST[$i]}
    NAME=${NAME_LIST[$i]}
    for EPOCHS in 5 7 10; do
        for BS in 16 32 64; do
            CUDA_VISIBLE_DEVICES=$N python linguistic_analysis/relation_classification.py \
                --model_name_or_path $MODEL_PATH \
                --do_train \
                --do_lower_case \
                --data_dir data/semeval2010_task8 \
                --eval_batch_size=32   \
                --train_batch_size=$BS   \
                --learning_rate=2e-5 \
                --num_train_epochs $EPOCHS \
                --output_dir linguistic_analysis/res/semeval/$NAME\_$EPOCHS\_$BS \
                --seed 0
        done
    done

done