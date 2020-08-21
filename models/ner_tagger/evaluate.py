import os, sys, json
from glob import glob
import numpy as np
from itertools import groupby, combinations
from functools import partial
from sklearn.metrics import cohen_kappa_score, precision_score, recall_score, precision_recall_fscore_support

def get_f1(p, r):
	return 2*p*r/(p+r)

def score_labels(true, pred, labels = None, average = 'micro', verbose = True):
	labels = labels or list(set(true))
	if verbose:
		print(true[:30])
		print(pred[:30])
		print(labels)

	prec = precision_score(true, pred, labels = labels, average = average)
	rec = recall_score(true, pred, labels = labels, average = average)
	f1 = get_f1(prec, rec)
	if verbose:
		print('f1        = %.2f' %f1)
		print('precision = %.2f' %prec)
		print('recall    = %.2f' %rec)
		class_ps = precision_score(true,pred,labels,average=None)
		class_rs = recall_score(true,pred,labels,average=None)
		for label, p, r in zip(labels, class_ps, class_rs):
			print('Label: %s' %label)
			print('\tf1        = %.2f' %get_f1(p, r))
			print('\tprecision = %.2f' %p)
			print('\trecall    = %.2f' %r)
		class_p = np.mean(class_ps)
		class_r = np.mean(class_rs)
		class_f1 = get_f1(class_p, class_r)
		print('avg class f1        = %.2f' %class_f1)
		print('avg class precision = %.2f' %class_p)
		print('avg class recall    = %.2f' %class_r)
	return { 'f1': f1, 'precision': prec, 'recall': rec }

def argmax(l):
	return l.index(max(l))

def thresh(l, x = 0.5):
	assert len(l) == 2
	return 1 if l[1] >= x else 0

def parse_sent_output(fname, label_fn):
	probs = [list(map(float, l.strip().split('\t'))) for l in open(fname).readlines()]
	labels = [label_fn(p) for p in probs]
	return labels

def parse_sent_input(fname):
	labels = [int(l.split('\t')[0]) for l in open(fname).readlines()]
	return labels

def score_sent_fname(input_fname, label_fn, ignore_0 = False, verbose = True):
	input_dirname, input_basename = os.path.split(input_fname)
	group = input_basename.split('.')[0]
	output_fname = '{}/results/{}_results.tsv'.format(input_dirname, group)
	
	true = parse_sent_input(input_fname)
	pred = parse_sent_output(output_fname, label_fn)

	labels = set(true)
	if ignore_0:
		labels.remove(0)
	return score_labels(true, pred, labels = list(labels), verbose = verbose)

def score_ner(fdir):
	output_fname = '{}/results/pred.txt'.format(fdir)
	results = json.load(open(output_fname))
	true = []
	pred = []
	for r in results:
		true += r['true_labels']
		pred += r['pred_labels']
	labels = set(true)
	labels.remove('0')
	score_labels(true, pred, labels = list(labels))

def sweep_thresh(input_fname):
	print('thsh | p     r     f1')
	for x in np.arange(0, 1.0, 0.05):
		s = score_input_fname(input_fname, partial(thresh, x=x), verbose = False)
		print('{:.2f} | {:.2f}  {:.2f}  {:.2f}'.format(x, s['precision'], s['recall'], s['f1']))

if __name__ == '__main__':
	score_ner(sys.argv[1])
