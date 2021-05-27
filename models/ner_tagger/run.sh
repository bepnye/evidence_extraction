#!/bin/bash
# current dir of this script

data_dir="/home/ben/Desktop/evidence_extraction/models/ner_tagger/data/ebm_o/"
output_dir="${data_dir}/results/"
model_dir="${data_dir}/model/"
#model_dir="/home/ben/Desktop/evidence_extraction/models/ner_tagger/data/ebm_nlp_ab3p/model/"
bert_base_dir="/home/ben/Desktop/biobert_pubmed/"

python bert_lstm_ner.py   \
        --do_lower_case=True \
        --do_train=True \
        --do_predict=True \
        --data_dir=$data_dir  \
        --model_dir=$model_dir \
        --output_dir=$output_dir \
        --label_idx=1 \
        --vocab_file=${bert_base_dir}/vocab.txt  \
        --bert_config_file=${bert_base_dir}/bert_config.json \
        --init_checkpoint=${bert_base_dir} \
        --data_config_path=$output_dir/data.conf \
