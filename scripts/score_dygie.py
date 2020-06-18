import json
import numpy as np
from collections import defaultdict
from sklearn.metrics import precision_recall_fscore_support

NEG_CLASS = 'NONE'

def overlap(x1, x2, y1, y2):
	return max(x1, y1) <= min(x2, y2)

class Doc:
	def __init__(self, tokens):
		self.text = ' '.join(tokens)
		self.tokens = tokens
		self.ner = [NEG_CLASS for _ in tokens]
		self.rel = {}
		self.entities = []

	def get_ner(self, ner_offsets):
		ner = [NEG_CLASS for _ in self.tokens]
		for i, f, l in ner_offsets:
			ner[i:f+1] = [l]*(f - i + 1)
		return ner

	def add_entities(self, clusters):
		for mention_offsets in clusters:
			eid = len(self.entities)
			self.entities.append(Entity(eid, self.tokens, mention_offsets))

	def get_overlapping_eids(self, i, f):
		return [e.id for e in self.entities if any([overlap(i, f, m.i, m.f) for m in e.mentions])]
	
	def get_rel(self, rel):
		# map from (eid1, eid2) => [relations...]
		# the target labels only have one relation per entity-pair, but the model
		# may predict different relations between each mention-mention pair
		e_rel = defaultdict(set)
		for m1_i, m1_f, m2_i, m2_f, r in rel:
			e1s = self.get_overlapping_eids(m1_i, m1_f)
			e2s = self.get_overlapping_eids(m2_i, m2_f)
			for e1 in e1s:
				for e2 in e2s:
					e_rel[(e1, e2)].add(r)
		return dict(e_rel)

	def comp_ner(self, pred_ner, average = 'micro', verbose = False):
		labels = ['i', 'o']
		if verbose:
			print('{}\t{}\t{}\t{}'.format('', 'p', 'r', 'f1'))
			for l in labels:
				p, r, f1, _ = precision_recall_fscore_support(self.ner, pred_ner, labels = [l], average = 'micro')
				print('{}\t{:.2f}\t{:.2f}\t{:.2f}'.format(l, p, r, f1))
			p, r, f1, _ = precision_recall_fscore_support(self.ner, pred_ner, labels = labels, average = average)
			print('{}\t{:.2f}\t{:.2f}\t{:.2f}'.format(average, p, r, f1))
		return self.ner, pred_ner

	def comp_rel(self, pred_rel, average = 'micro', verbose = False):
		true = []
		pred = []
		# score the model on each of the true relations between entities
		# this will cover all of the TPs and FNs
		for (e1, e2), rs in self.rel.items():
			# gold-standard should only have one relation between entities
			assert len(rs) == 1
			true_l = list(rs)[0]
			pred_ls = pred_rel.get((e1, e2), {NEG_CLASS})
			for pred_l in list(pred_ls):
				true.append(true_l)
				pred.append(pred_l)
		# finally, count all the FPs (predicted e1-e2 relations that don't exist)
		for (e1, e2), rs in pred_rel.items():
			# already counted these
			if (e1, e2) in self.rel: continue
			for pred_l in list(rs):
				true.append(NEG_CLASS)
				pred.append(pred_l)
		labels = ['SAME', 'DECR', 'INCR']
		if verbose:
			print('{}\t{}\t{}\t{}'.format('', 'p', 'r', 'f1'))
			for l in labels:
				p, r, f1, _ = precision_recall_fscore_support(true, pred, labels = [l], average = 'micro')
				print('{}\t{:.2f}\t{:.2f}\t{:.2f}'.format(l, p, r, f1))
			p, r, f1, _ = precision_recall_fscore_support(true, pred, labels = labels, average = average)
			print('{}\t{:.2f}\t{:.2f}\t{:.2f}'.format(average, p, r, f1))
		return true, pred

class Entity:
	def __init__(self, id_, tokens, mention_offsets):
		self.id = id_
		self.mentions = [Mention(tokens, i, f) for i, f in mention_offsets]
		self.text = self.mentions[0].text

class Mention:
	def __init__(self, tokens, i, f):
		self.i = i
		self.f = f
		self.text = ' '.join(tokens[i:f+1])

def load_inputs(fname):
	rows = [json.loads(l.strip()) for l in open(fname).readlines()]
	docs = {}
	for r in rows:
		assert len(r['sentences']) == 1
		doc = Doc(r['sentences'][0])
		doc.add_entities(r['clusters'])
		doc.ner = doc.get_ner(r['ner'][0])
		doc.rel = doc.get_rel(r['relations'][0])
		docs[r['doc_key']] = doc
	return docs

def print_label_scores(label_pairs, doc_micro = True):
	if doc_micro:
		per_doc_t, per_doc_p = zip(*label_pairs)
		true = sum(per_doc_t, [])
		pred = sum(per_doc_p, [])
		labels = set(true) - {NEG_CLASS}
		print('micro-averaged across docs')
		print('{}\t{}\t{}\t{}'.format('', 'p', 'r', 'f1'))
		for l in labels:
			p, r, f1, _ = precision_recall_fscore_support(true, pred, labels = [l], average = 'micro')
			print('{}\t{:.2f}\t{:.2f}\t{:.2f}'.format(l, p, r, f1))
		p, r, f1, _ = precision_recall_fscore_support(true, pred, labels = list(labels), average = 'micro')
		print('{}\t{:.2f}\t{:.2f}\t{:.2f}'.format('micro', p, r, f1))
	else:
		doc_scores = []
		for true, pred in label_pairs:
			labels = set(true) - {NEG_CLASS}
			doc_scores.append(precision_recall_fscore_support(true, pred, labels = list(labels), average = 'micro'))
		print('macro-averaged across docs')
		ps, rs, f1s, supp = zip(*doc_scores)
		print('{}\t{}\t{}\t{}'.format('', 'p', 'r', 'f1'))
		print('{}\t{:.2f}\t{:.2f}\t{:.2f}'.format('macro', np.mean(ps), np.mean(rs), np.mean(f1s)))

def score_outputs(docs, fname):
	rows = [json.loads(l.strip()) for l in open(fname).readlines()]
	assert len(rows) == len(docs)
	ner_labels = []
	rel_labels = []
	for r in rows:
		d = docs[r['doc_key']]
		ner = d.get_ner(r['predicted_ner'][0])
		ner_labels.append(d.comp_ner(ner))
		rel = d.get_rel(r['predicted_relations'][0])
		rel_labels.append(d.comp_rel(rel))
	print_label_scores(ner_labels) 
	print()
	print_label_scores(rel_labels)


def score_model(input_fname, output_fname = None):
	docs = load_inputs(input_fname)
	output_fname = output_fname or input_fname.replace('_input', '')
	score_outputs(docs, output_fname)
