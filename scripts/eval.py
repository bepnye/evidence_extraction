import sys
from glob import glob
from itertools import groupby, combinations
from sklearn.metrics import cohen_kappa_score, precision_score, recall_score, precision_recall_fscore_support
  
def get_f1(prec, rec):
  return 2*prec*rec/(prec+rec)

def token_f1(true, pred, labels):

  print(true[:30])
  print(pred[:30])
  print(labels)

  prec = precision_score(true, pred, labels = labels, average='micro')
  rec = recall_score(true, pred, labels = labels, average='micro')
  f1 = get_f1(prec, rec)
  print('f1        = %.2f' %f1)
  print('precision = %.2f' %prec)
  print('recall    = %.2f' %rec)
  class_scores = zip(labels, precision_score(true,pred,labels,average=None), recall_score(true,pred,labels,average=None))
  for label, prec, rec in class_scores:
    print('Label: %s' %label)
    print('\tf1        = %.2f' %get_f1(prec, rec))
    print('\tprecision = %.2f' %prec)
    print('\trecall    = %.2f' %rec)
  return { 'f1': f1, 'precision': prec, 'recall': rec }

def eval_bert_probs(data_dir):
  dev_probs = [[float(x) for x in l.strip().split('\t')] \
      for l in open('{}/results/eval_results.tsv'.format(data_dir)).readlines()]
  dev_pred = [probs.index(max(probs)) for probs in dev_probs]
  dev_true = [int(l.split('\t')[0]) for l in open('{}/dev.tsv'.format(data_dir)).readlines()]

  token_f1(dev_true, dev_pred, list(set(dev_true)))

if __name__ == '__main__':
  eval_bert_probs(sys.argv[1])
