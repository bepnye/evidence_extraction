lowercase='False'
bert_model_dir=/home/ben/Desktop/biobert_pubmed/

python bert_lstm_ner.py   \
        --task_name="NER"  \
        --do_train=False   \
        --use_feature_based=False \
        --do_predict=True \
        --use_crf=True \
        --data_dir=${CDIR}/data/ico/diff_same_neg/ \
        --label_idx=0 \
        --vocab_file=${bert_model_dir}/vocab.txt  \
        --do_lower_case=${lowercase} \
        --bert_config_file=${bert_model_dir}/bert_config.json \
        --max_seq_length=150   \
        --lstm_size=256 \
        --eval_batch_size=32   \
        --predict_batch_size=32   \
        --allow_unk_label=True \
        --data_config_path=${CDIR}/data.conf \
        --output_dir=${CDIR}/output/result_dir

#perl conlleval.pl < ${CDIR}/output/result_dir/pred.txt
