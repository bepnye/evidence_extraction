BERT_BASE_DIR=/home/ben/Desktop/scibert_scivocab_uncased
DATA_DIR='data/ev_sent_fulltext/'
OUTPUT_DIR=${DATA_DIR}/results
MODEL_DIR=${DATA_DIR}/model

python run_classifier.py \
  --task_name=ico \
  --do_train=True \
  --do_train_eval=False \
  --do_eval=True \
  --do_predict=False \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --max_seq_length=256 \
  --train_batch_size=16 \
  --learning_rate=1e-5 \
  --num_train_epochs=3.0 \
  --data_dir=$DATA_DIR \
  --model_dir=$MODEL_DIR \
  --output_dir=$OUTPUT_DIR
