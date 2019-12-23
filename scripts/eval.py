import sys, os
from glob import glob
from itertools import groupby, combinations
from sklearn.metrics import cohen_kappa_score, precision_score, recall_score, precision_recall_fscore_support
  
def get_f1(prec, rec):
  return 2*prec*rec/(prec+rec)

def token_f1(true, pred, labels):
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

def eval_token_labels(pred_dir, ebm_nlp_top = 'ebm_nlp_2_00/'):
  for e in ['participants', 'interventions', 'outcomes']:
    print('Computing score for {}'.format(e))
    class_pred_dir = '{}/{}/'.format(pred_dir, e)
    class_test_dir = '{}/annotations/aggregated/starting_spans/{}/test/gold/'.format(ebm_nlp_top, e)
    eval_token_class_labels(class_test_dir, class_pred_dir)
    print()

def eval_token_class_labels(test_dir, pred_dir):
  true_fnames = glob('{}/*.ann'.format(test_dir))
  all_pred_labels = []
  all_true_labels = []
  for true_fname in true_fnames:
    pmid, worker, sfx = os.path.basename(true_fname).split('.')
    pred_fname = '{}/{}.AGGREGATED.ann'.format(pred_dir, pmid)
    true_labels = open(true_fname).read().strip().split('\n')
    if not os.path.isfile(pred_fname):
      print('Unable to find pred labels for [{}] at [{}]'.format(pmid, pred_fname))
      pred_labels = ['0']*len(true_labels)
    else:
      pred_labels = open(pred_fname).read().strip().split('\n')
    all_pred_labels += pred_labels
    all_true_labels += true_labels
  token_f1(all_true_labels, all_pred_labels, ['1'])

def eval_bert_probs(data_dir):
  dev_probs = [[float(x) for x in l.strip().split('\t')] \
      for l in open('{}/results/eval_results.tsv'.format(data_dir)).readlines()]
  dev_pred = [probs.index(max(probs)) for probs in dev_probs]
  dev_true = [int(l.split('\t')[0]) for l in open('{}/dev.tsv'.format(data_dir)).readlines()]

  token_f1(dev_true, dev_pred, list(set(dev_true)))

def eval_bert_ner(data_dir):
  lines = [l.strip() for l in open('{}/results/pred.txt'.format(data_dir)).readlines()]
  lines = [l for l in lines if l]
  true, pred, tokens = zip(*[l.split() for l in lines])
  token_f1(true, pred, list(set(true) - set('N')))

if __name__ == '__main__':
  #eval_bert_probs(sys.argv[1])
  eval_bert_ner(sys.argv[1])
