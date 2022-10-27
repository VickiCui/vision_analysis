#
# Usage: ./get_sentence_data.sh
#

set -e

CORPUS_PATH=./data/CSLB_Property_Norms_V1.1
mkdir -p $CORPUS_PATH

DUMP_NAME=norms.dat
SAVE_NAME=cslb.json
MAIN_PATH=$CORPUS_PATH

if [ ! -f $CORPUS_PATH/cslb.json ]; then
    echo "Pre-processing CSLB corpus ..."
    python prepare_CSLB_pair.py --data_path $CORPUS_PATH/$DUMP_NAME --save_path $CORPUS_PATH/$SAVE_NAME
    echo "Converted $DUMP_NAME in $CORPUS_PATH/$SAVE_NAME (train/val)"
fi