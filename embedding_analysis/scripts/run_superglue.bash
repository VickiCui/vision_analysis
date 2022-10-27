N=1
MT_LIST=(bert bert bert bert bert)
MODEL_PATH_LIST=(pretrained_models/bert_voken pretrained_models/voken_base pretrained_models/vidlankd pretrained_models/bert_wiki_coco pretrained_models/bert_wiki_howto100m)
NAME_LIST=(bert bert-i bert-v voken vidlankd)

for(( i=0;i<${#MODEL_PATH_LIST[@]};i++)); do
    MODEL_PATH=${MODEL_PATH_LIST[$i]}
    NAME=${NAME_LIST[$i]}
    for BS in 64; do
        for EPOCH in 3; do
            for TASK in wic; do
                CUDA_VISIBLE_DEVICES=$N python linguistic_analysis/superglue.py \
                    --model_name_or_path $MODEL_PATH \
                    --model_name $NAME \
                    --data_dir data/superglue \
                    --task $TASK \
                    --batch_size $BS \
                    --epoch $EPOCH \
                    --run_name $NAME \
                    --learning_rate 5e-5
            done
        done
    done
done