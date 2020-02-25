#!/bin/bash
# current dir of this script
ts_chunk=$1
data_dir="${ts_chunk}/ner/"
output_dir="${ts_chunk}/ner/results/"

model_dir="/home/ben/Desktop/evidence_extraction/models/ner_tagger/data/ebm_nlp_ab3p/model/"

bert_base_dir="/home/ben/Desktop/biobert_pubmed/"

python bert_lstm_ner.py   \
	--visible_devices="0" \
        --task_name="ner"  \
        --do_train=False \
        --do_predict=True \
        --collapse_wp=True \
        --use_feature_based=False \
        --use_crf=True \
        --data_dir=$data_dir  \
        --model_dir=$model_dir \
        --output_dir=$output_dir \
        --label_idx=1 \
        --vocab_file=${bert_base_dir}/vocab.txt  \
        --do_lower_case=True \
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
