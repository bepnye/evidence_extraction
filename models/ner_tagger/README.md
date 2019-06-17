# NER evaluation for CoNLL 2003

## Modification

- modify the project for CoNLL 2003 data.
- add train.sh, predict.sh
- add multi-layered fused_lstm_layer() which uses LSTMBlockFusedCell.
- add tf.train.LoggingTensorHook for printing loss while training.
- add tf.estimator.train_and_evaluate() with stop_if_no_increase_hook()

## How to train and predict

### download BERT model
```
* download BERT model

$ ls cased_L-12_H-768_A-12 uncased_L-12_H-768_A-12
cased_L-12_H-768_A-12:
bert_config.json  bert_model.ckpt.data-00000-of-00001  bert_model.ckpt.index  bert_model.ckpt.meta  vocab.txt

uncased_L-12_H-768_A-12:
bert_config.json  bert_model.ckpt.data-00000-of-00001  bert_model.ckpt.index  bert_model.ckpt.meta  vocab.txt
```

### train
```
* edit 'bert_model_dir'
* edit 'lowercase=False' for cased BERT model, 'lowercase=True' for uncased.
$ ./train.sh -v -v
$ tensorboard --logdir output/result_dir/ --port 6008
```
![](/images/eval_f.png)
![](/images/loss.png)

### predict
```
* select the best model among output/result_dir and edit 'output/result_dir/checkpoint' file.
$ ./predict.sh -v -v

$ cat output/result_dir/predicted_results.txt

$ more output/result_dir/pred.txt
...
Nadim NNP B-NP B-PER B-PER
Ladki NNP I-NP I-PER I-PER

AL-AIN NNP B-NP B-LOC B-LOC
, , O O O
United NNP B-NP B-LOC B-LOC
Arab NNP I-NP I-LOC I-LOC
Emirates NNPS I-NP I-LOC I-LOC
1996-12-06 CD I-NP O O
...

$ perl conlleval.pl < output/result_dir/pred.txt
* base, cased
processed 46435 tokens with 5648 phrases; found: 5637 phrases; correct: 5163.
accuracy:  98.30%; precision:  91.59%; recall:  91.41%; FB1:  91.50
              LOC: precision:  93.27%; recall:  92.27%; FB1:  92.77  1650
             MISC: precision:  81.01%; recall:  82.62%; FB1:  81.81  716
              ORG: precision:  89.85%; recall:  90.61%; FB1:  90.23  1675
              PER: precision:  96.43%; recall:  95.18%; FB1:  95.80  1596
processed 46435 tokens with 5648 phrases; found: 5675 phrases; correct: 5183.
accuracy:  98.32%; precision:  91.33%; recall:  91.77%; FB1:  91.55
              LOC: precision:  92.95%; recall:  92.45%; FB1:  92.70  1659
             MISC: precision:  82.50%; recall:  82.62%; FB1:  82.56  703
              ORG: precision:  88.37%; recall:  91.45%; FB1:  89.88  1719
              PER: precision:  96.74%; recall:  95.36%; FB1:  96.04  1594

* large, cased
processed 46435 tokens with 5648 phrases; found: 5663 phrases; correct: 5212.
accuracy:  98.47%; precision:  92.04%; recall:  92.28%; FB1:  92.16
              LOC: precision:  93.18%; recall:  93.35%; FB1:  93.26  1671
             MISC: precision:  83.53%; recall:  82.34%; FB1:  82.93  692
              ORG: precision:  90.57%; recall:  91.39%; FB1:  90.98  1676
              PER: precision:  96.00%; recall:  96.41%; FB1:  96.20  1624

* large, cased, feature-based
processed 46435 tokens with 5648 phrases; found: 5680 phrases; correct: 5195.
accuracy:  98.33%; precision:  91.46%; recall:  91.98%; FB1:  91.72
              LOC: precision:  92.55%; recall:  93.05%; FB1:  92.80  1677
             MISC: precision:  83.16%; recall:  81.62%; FB1:  82.39  689
              ORG: precision:  89.54%; recall:  91.75%; FB1:  90.63  1702
              PER: precision:  95.91%; recall:  95.61%; FB1:  95.76  1612

* large, cased, -2 layer
processed 46435 tokens with 5648 phrases; found: 5669 phrases; correct: 5194.
accuracy:  98.34%; precision:  91.62%; recall:  91.96%; FB1:  91.79
              LOC: precision:  94.39%; recall:  91.79%; FB1:  93.07  1622
             MISC: precision:  82.88%; recall:  82.76%; FB1:  82.82  701
              ORG: precision:  87.69%; recall:  92.23%; FB1:  89.91  1747
              PER: precision:  96.94%; recall:  95.86%; FB1:  96.39  1599
```

### dev note
```
1. dev.txt

- BERT

* base, cased
INFO:tensorflow:Saving dict for global step 30000: eval_accuracy = 0.9934853, eval_f = 0.9627948, eval_loss = 1.6617825, eval_precision = 0.9645357, eval_recall = 0.9610601, global_step = 30000, loss = 1.6632456

* large, cased
INFO:tensorflow:Saving dict for global step 31000: eval_accuracy = 0.9936458, eval_f = 0.96526873, eval_loss = 1.6502532, eval_precision = 0.9670967, eval_recall = 0.9634477, global_step = 31000, loss = 1.6502532

- ELMo

[epoch 33/70] dev precision, recall, f1(token):
precision, recall, fscore
[0.9978130380159136, 0.9583333333333334, 0.9263271939328277, 0.9706510138740662, 0.9892224788298691, 0.9701897018970189, 0.9464285714285714, 0.8797653958944281, 0.9323308270676691, 0.959865053513262]
[0.9979755671902268, 0.9433258762117822, 0.9273318872017353, 0.987513572204126, 0.9831675592960979, 0.9744148067501361, 0.9174434087882823, 0.8670520231213873, 0.9649805447470817, 0.9590840404510055]
[0.9978942959852019, 0.9507703870725291, 0.9268292682926829, 0.9790096878363832, 0.9861857252494244, 0.9722976643128735, 0.9317106152805951, 0.8733624454148471, 0.9483747609942639, 0.9594743880458168]
new best f1 score! : 0.9594743880458168
max model saved in file: ./checkpoint/model_max.ckpt


2. test.txt

- BERT

* base, cased
INFO:tensorflow:Saving dict for global step 30000: eval_accuracy = 0.9861725, eval_f = 0.92653006, eval_loss = 3.263393, eval_precision = 0.9218941, eval_recall = 0.931213, global_step = 30000, loss = 3.263018

* large, cased
INFO:tensorflow:Saving dict for global step 31000: eval_accuracy = 0.98725253, eval_f = 0.9329729, eval_loss = 3.080433, eval_precision = 0.9299449, eval_recall = 0.93602073, global_step = 31000, loss = 3.0799496


3. word-based BERT

- model download : https://github.com/dsindex/bert

* base, uncased

model : engwiki.1m-step.uncased_L-12_H-768_A-12
lstm_size : 256

processed 46435 tokens with 5648 phrases; found: 5669 phrases; correct: 4985.
accuracy:  97.45%; precision:  87.93%; recall:  88.26%; FB1:  88.10
              LOC: precision:  90.45%; recall:  91.37%; FB1:  90.90  1685
             MISC: precision:  75.59%; recall:  77.21%; FB1:  76.39  717
              ORG: precision:  85.42%; recall:  83.56%; FB1:  84.48  1625
              PER: precision:  93.24%; recall:  94.68%; FB1:  93.96  1642

```

----

# README from source git

Tensorflow solution of NER task Using BiLSTM-CRF model with Google BERT Fine-tuning

使用谷歌的BERT模型在BLSTM-CRF模型上进行预训练用于中文命名实体识别的Tensorflow代码'

Welcome to star this repository!

The Chinese training data($PATH/NERdata/) come from:https://github.com/zjy-ucas/ChineseNER 
  
The CoNLL-2003 data($PATH/NERdata/ori/) come from:https://github.com/kyzhouhzau/BERT-NER 
  
The evaluation codes come from:https://github.com/guillaumegenthial/tf_metrics/blob/master/tf_metrics/__init__.py  


Try to implement NER work based on google's BERT code and BiLSTM-CRF network!


## How to train

#### 1.using config param in terminal

```
  python3 bert_lstm_ner.py   \
                  --task_name="NER"  \ 
                  --do_train=True   \
                  --do_eval=True   \
                  --do_predict=True
                  --data_dir=NERdata   \
                  --vocab_file=checkpoint/vocab.txt  \ 
                  --bert_config_file=checkpoint/bert_config.json \  
                  --init_checkpoint=checkpoint/bert_model.ckpt   \
                  --max_seq_length=128   \
                  --train_batch_size=32   \
                  --learning_rate=2e-5   \
                  --num_train_epochs=3.0   \
                  --output_dir=./output/result_dir/ 
 ```       
 #### 2. replace the BERT path and project path in bert_lstm_ner.py.py
 ```
 if os.name == 'nt':
    bert_path = '{your BERT model path}'
    root_path = '{project path}'
else:
    bert_path = '{your BERT model path}'
    root_path = '{project path}'
 ```

## result:
all params using default
#### In dev data set:
![](/images/picture1.png)

#### In test data set
![](/images/picture2.png)

## reference: 
+ The evaluation codes come from:https://github.com/guillaumegenthial/tf_metrics/blob/master/tf_metrics/__init__.py

+ [https://github.com/google-research/bert](https://github.com/google-research/bert)
      
+ [https://github.com/kyzhouhzau/BERT-NER](https://github.com/kyzhouhzau/BERT-NER)

+ [https://github.com/zjy-ucas/ChineseNER](https://github.com/zjy-ucas/ChineseNER)

> Any problem please email me(ma_cancan@163.com)
