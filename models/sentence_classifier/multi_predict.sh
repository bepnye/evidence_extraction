BERT_BASE_DIR=/home/ben/Desktop/biobert_pubmed
MODEL_DIR=/home/ben/Desktop/evidence_extraction/data/sent_classifier/ev_binary/model/
DATA_DIR=/home/ben/Desktop/evidence_extraction/data/sent_classifier/trialstreamer/0_9999/
OUTPUT_DIR=${DATA_DIR}/results

python run_classifier.py \
  --task_name=ico_pred \
  --do_train=false \
  --do_eval=true \
  --do_predict=false \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/biobert_model.ckpt \
  --max_seq_length=150 \
  --train_batch_size=16 \
  --learning_rate=2e-5 \
  --num_train_epochs=3.0 \
  --data_dir=$DATA_DIR \
  --model_dir=$MODEL_DIR \
  --output_dir=$OUTPUT_DIR
