N=2


MT_LIST=(bert bert bert bert bert)
MODEL_PATH_LIST=(pretrained_models/bert_voken pretrained_models/voken_base pretrained_models/vidlankd pretrained_models/bert_wiki_coco pretrained_models/bert_wiki_howto100m)
NAME_LIST=(bert bert-i bert-v voken vidlankd)


for(( i=0;i<${#MT_LIST[@]};i++)); do
    TYPE=${MT_LIST[$i]}
    MODEL_PATH=${MODEL_PATH_LIST[$i]}
    NAME=${NAME_LIST[$i]}

    for EPOCHS in 5 7 10; do
        for BS in 32 64 128; do
            for LR in 1e-3 5e-3 5e-4; do
                CUDA_VISIBLE_DEVICES=$N python linguistic_analysis/sclass.py \
                    --model_name_or_path $MODEL_PATH \
                    --do_train \
                    --do_eval \
                    --do_test \
                    --do_lower_case \
                    --data_dir data/WIKI-PSE/dataset \
                    --cache_dir data/WIKI-PSE/cache_dataset/$NAME \
                    --eval_batch_size=32   \
                    --train_batch_size=$BS   \
                    --learning_rate=$LR \
                    --num_train_epochs $EPOCHS.0 \
                    --output_dir linguistic_analysis/res/wiki_sclass/$NAME\_$EPOCHS\_$BS\_$LR \
                    --softmax
            done
        done
    done
done

for(( i=0;i<${#MT_LIST[@]};i++)); do
    TYPE=${MT_LIST[$i]}
    MODEL_PATH=${MODEL_PATH_LIST[$i]}
    NAME=${NAME_LIST[$i]}
    for EPOCHS in 7 10 13 16; do
        for BS in 64 128 256; do
            for LR in 5e-3 1e-2; do
                for SEED in 0 2021 23333; do
                    CUDA_VISIBLE_DEVICES=$N python linguistic_analysis/sclass.py \
                        --model_name_or_path $MODEL_PATH \
                        --do_train \
                        --do_eval \
                        --do_test \
                        --do_lower_case \
                        --data_dir data/SEMCATdataset2018 \
                        --cache_dir data/SEMCATdataset2018/cache_dataset/$NAME \
                        --eval_batch_size=32   \
                        --train_batch_size=$BS   \
                        --learning_rate=$LR \
                        --num_train_epochs $EPOCHS.0 \
                        --output_dir linguistic_analysis/res/semcat_sclass/$NAME\_$EPOCHS\_$BS\_$LR\_$SEED \
                        --softmax \
                        --seed $SEED
                done
            done
        done
    done
done
