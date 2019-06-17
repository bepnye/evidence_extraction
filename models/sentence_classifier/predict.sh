DATA_DIR=../..//data/sent_classifier/ev_trinary
OUTPUT_DIR=${DATA_DIR}/results
MODEL_DIR=${DATA_DIR}/model

python run_classifier.py \
  --task_name=ico_pred \
  --do_train=false \
  --do_eval=true \
  --do_predict=false \
  --max_seq_length=150 \
  --train_batch_size=16 \
  --learning_rate=2e-5 \
  --num_train_epochs=3.0 \
  --data_dir=$DATA_DIR \
  --model_dir=$MODEL_DIR \
  --output_dir=$OUTPUT_DIR
