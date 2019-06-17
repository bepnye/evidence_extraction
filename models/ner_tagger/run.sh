#!/bin/bash
# current dir of this script
CDIR=$(readlink -f $(dirname $(readlink -f ${BASH_SOURCE[0]})))

data_prefix="p1_all"
data_dir="data/${data_prefix}/"
model_dir="data/${data_prefix}/model/"
output_dir="data/${data_prefix}/results/"

train=1

if [ $train == 1 ]; then
    do_train=True
    do_predict=False
    read -p "Clearing output_dir $output_dir, [y] to proceed: " ok
    case ${ok:0:1} in
        y|Y )
            echo "    proceeding"
            rm -rf $output_dir/*
            mkdir -p $output_dir
        ;;
        * )
            echo "    aborting"
            exit 1
        ;;
    esac
else
    do_train=False
    do_predict=True
fi

bert_base_dir=/home/ben/Desktop/biobert_pubmed/
python bert_lstm_ner.py   \
	--visible_devices="0" \
        --task_name="NER"  \
        --do_train=$do_train \
        --do_predict=$do_predict \
        --collapse_wp=True \
        --use_feature_based=False \
        --use_crf=True \
        --data_dir=$data_dir  \
        --model_dir=$model_dir \
        --output_dir=$output_dir \
        --label_idx=1 \
        --vocab_file=${bert_base_dir}/vocab.txt  \
        --do_lower_case=False \
        --bert_config_file=${bert_base_dir}/bert_config.json \
        --init_checkpoint=${bert_base_dir} \
        --max_seq_length=150   \
        --lstm_size=256 \
        --train_batch_size=16   \
        --eval_batch_size=32   \
        --predict_batch_size=32   \
        --bert_dropout_rate=0.2 \
        --bilstm_dropout_rate=0.5 \
        --learning_rate=2e-5   \
        --num_train_epochs=10   \
        --data_config_path=$output_dir/data.conf \
