import sys, random, json, re, glob, os
import itertools
import pickle as pkl
from collections import defaultdict, Counter
from operator import itemgetter
from functools import partial
import traceback

from bert_serving.client import BertClient
BC = None
#from stanfordnlp import Pipeline
#SF = None
from scipy.spatial.distance import cosine as cos_dist
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering as agg_cluster
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import numpy as np

from coref_scores import b_cubed, muc, ceaf_e
from ner_scores import compute_metrics as ner_score
import classes
import utils

# wrapper so we can use non-BC functions without starting the client
def encode(str_list):
	global BC
	if BC is None:
		BC = BertClient()
	return BC.encode(str_list)

"""

NER preprocessing
Adds spans to doc.labels['NER_*']

"""
NER_LABEL_MAP = { \
		'I_int': 'i',
		'I_out': 'o',
		'i': 'i',
		'o': 'o',
		'p': 'p',
}

def add_ic_ev_output(docs, group, fdir = '../models/sentence_classifier/data/i_c_intro'):
	model_input = '{}/{}.tsv'.format(fdir, group)
	model_output = '{}/results/{}_results.tsv'.format(fdir, group)
	inputs = [l.strip().split('\t') for l in open(model_input).readlines()]
	outputs = [list(map(float, l.strip().split('\t'))) for l in open(model_output).readlines()]
	assert len(inputs) == len(outputs)
	pmid_ev_map = defaultdict(lambda: defaultdict(list))
	for (_, pmid, ev_i, ev_f, i_i, i_f, i_text, context), class_probs in zip(inputs, outputs):
		result = { \
				'class_probs': list(map(float, class_probs)),
				'idx_i': int(i_i),
				'idx_f': int(i_f),
				'text': i_text
		}
		pmid_ev_map[pmid][(int(ev_i), int(ev_f))].append(result)
	for doc in docs:
		for (ev_i, ev_f), results in pmid_ev_map[doc.id].items():
			sents = [s for s in doc.labels['BERT_ev'] if s.i == ev_i and s.f == ev_f]
			assert len(sents) == 1
			sent = sents[0]
			if len(results) == 0:
				sent.pred_i = classes.Span(-1, -1, 'Invalid Span')
			best_i = max(results, key = lambda r: r['class_probs'][2])
			best_c = max(results, key = lambda r: r['class_probs'][1])
			sent.pred_i = classes.Span(best_i['idx_i'], best_i['idx_f'], best_i['text'])
			sent.pred_c = classes.Span(best_c['idx_i'], best_c['idx_f'], best_c['text'])
			try:
				assert sent.pred_i.text == utils.clean_str(doc.text[sent.pred_i.i:sent.pred_i.f])
			except AssertionError:
				print('Mismatch for I when loading IC results...')
				print(sent.pred_i.text)
				print(utils.clean_str(doc.text[sent.pred_i.i:sent.pred_i.f]))
			try:
				assert sent.pred_c.text == utils.clean_str(doc.text[sent.pred_c.i:sent.pred_c.f])
			except AssertionError:
				print('Mismatch for C when loading IC results...')
				print(sent.pred_c.text)
				print(utils.clean_str(doc.text[sent.pred_c.i:sent.pred_c.f]))
			sent.pred_os = utils.s_overlaps(sent, doc.labels['NER_o'])
		for ev in doc.labels['BERT_ev']:
			# if there were no NER_i spans to chose from, the argmax won't even fire - fill in a default
			if not hasattr(ev, 'pred_i'):
				assert not hasattr(ev, 'pred_c')
				ev.pred_i = classes.Span(-1, -1, 'Invalid Span')
				ev.pred_c = classes.Span(-1, -1, 'Invalid Span')
			if not hasattr(ev, 'pred_os'):
				ev.pred_os = []

def get_overlapping_entities(doc, s):
	return [e.name for e in doc.entities if any(utils.s_overlaps(s, e.mentions))]

def eval_frame_linking(docs):
	o_links = []
	i_links = []
	c_links = []
	for d in docs:
		for f in d.frames:
			evs = [s for s in d.labels['BERT_ev'] if s.i == f.ev.i and s.f == f.ev.f]
			assert len(evs) == 1
			ev = evs[0]
			gold_i_spans = d.labels['GOLD_{}'.format(f.i.label)]
			pred_i = ev.pred_i
			i_links.append(any(utils.s_overlaps(pred_i, gold_i_spans)))

			gold_c_spans = d.labels['GOLD_{}'.format(f.c.label)]
			pred_c = ev.pred_c
			c_links.append(any(utils.s_overlaps(pred_c, gold_c_spans)))

			gold_o_spans = d.labels['GOLD_{}'.format(f.o.label)]
			found_o = False
			for pred_o in ev.pred_os:
				if any(utils.s_overlaps(pred_o, gold_o_spans)):
					found_o = True
					break
			o_links.append(found_o)
	print('i: {:.2f}'.format(np.mean(i_links)))
	print('c: {:.2f}'.format(np.mean(c_links)))
	print('o: {:.2f}'.format(np.mean(o_links)))

NEG_CLASS = 'NULL'
def eval_doc_relations(doc, verbose = False):
	pred_rels = defaultdict(list)
	for ev in doc.labels['BERT_ev']:
		i_entities = get_overlapping_entities(doc, ev.pred_i)
		c_entities = get_overlapping_entities(doc, ev.pred_c)
		for pred_o in ev.pred_os:
			o_entities = get_overlapping_entities(doc, pred_o)
			rel = get_frame_rel(pred_o.label)
			for i, o in itertools.product(i_entities, o_entities):
				pred_rels[i,o].append(rel)
	if verbose:
		print_frames(doc)
	true = []
	pred = []
	# score the model on each of the true relations between entities
	# this will cover all of the TPs and FNs
	for (e1, e2), true_l in doc.relations.items():
		if true_l == NEG_CLASS: continue
		pred_ls = pred_rels.get((e1, e2), [NEG_CLASS])
		for pred_l in list(pred_ls):
			print('{} {} [{}] [{}]'.format(true_l, pred_l, e1, e2))
			true.append(true_l)
			pred.append(pred_l)
	# finally, count all the FPs (predicted e1-e2 relations that don't exist)
	for (e1, e2), rs in pred_rels.items():
		# already counted these
		if (e1, e2) in doc.relations: continue
		for pred_l in list(rs):
			true.append(NEG_CLASS)
			pred.append(pred_l)
			print('{} {} [{}] [{}]'.format(NEG_CLASS, pred_l, e1, e2))
	labels = ['SAME', 'DECR', 'INCR']
	if verbose:
		print('{}\t{}\t{}\t{}'.format('', 'p', 'r', 'f1'))
		for l in labels:
			p, r, f1, _ = precision_recall_fscore_support(true, pred, labels = [l], average = 'micro')
			print('{}\t{:.2f}\t{:.2f}\t{:.2f}'.format(l, p, r, f1))
		p, r, f1, _ = precision_recall_fscore_support(true, pred, labels = labels, average = 'micro')
		print('{}\t{:.2f}\t{:.2f}\t{:.2f}'.format('micro', p, r, f1))
	return true, pred

def eval_relations(docs):
	true = []
	pred = []
	for d in docs:
		t, p = eval_doc_relations(d)
		true += t
		pred += p
	labels = ['SAME', 'DECR', 'INCR']
	print('{}\t{}\t{}\t{}'.format('', 'p', 'r', 'f1'))
	for l in labels:
		p, r, f1, _ = precision_recall_fscore_support(true, pred, labels = [l], average = 'micro')
		print('{}\t{:.2f}\t{:.2f}\t{:.2f}'.format(l, p, r, f1))
	p, r, f1, _ = precision_recall_fscore_support(true, pred, labels = labels, average = 'micro')
	print('{}\t{:.2f}\t{:.2f}\t{:.2f}'.format('micro', p, r, f1))
	print(confusion_matrix(true, pred, labels = labels + [NEG_CLASS]))

def add_ev_sent_output(docs, group, fdir, label_fn = utils.argmax):
	model_input = '{}/{}.tsv'.format(fdir, group)
	model_output = '{}/results/{}_results.tsv'.format(fdir, group)
	inputs = [l.strip().split('\t') for l in open(model_input).readlines()]
	outputs = [l.strip().split('\t') for l in open(model_output).readlines()]
	assert len(inputs) == len(outputs)
	pmid_sent_labels = defaultdict(list)
	for (true_label, pmid, sent), class_probs in zip(inputs, outputs):
		label = label_fn(list(map(float, class_probs)))
		pmid_sent_labels[pmid].append(label)
	for doc in docs:
		sent_labels = pmid_sent_labels[doc.id]
		try:
			assert len(sent_labels) == len(doc.sents)
		except AssertionError:
			print('Unable to match sents to outputs for pmid={}'.format(doc.id))
			print(doc.id, len(sent_labels), len(doc.sents))
			continue
		doc.labels['BERT_ev'] = [s for s, l in zip(doc.sents, sent_labels) if l]

def add_o_ev_output(docs, group, fdir = '../models/sentence_classifier/data/o_ev_sent'):
	model_input = '{}/{}.tsv'.format(fdir, group)
	model_output = '{}/results/{}_results.tsv'.format(fdir, group)
	inputs = [l.strip().split('\t') for l in open(model_input).readlines()]
	outputs = [list(map(float, l.strip().split('\t'))) for l in open(model_output).readlines()]
	assert len(inputs) == len(outputs)
	pmid_offset_labels = defaultdict(dict)
	for (_, pmid, i, f, o_text, ev_text), class_probs in zip(inputs, outputs):
		label = utils.argmax(class_probs) - 1 # [0,1,2] => [-1,0,1]
		pmid_offset_labels[pmid][(int(i), int(f))] = label
	for doc in docs:
		offset_labels = pmid_offset_labels[doc.id]
		for o_span in doc.labels['NER_o']:
			k = (o_span.i, o_span.f)
			if k in offset_labels: # k in offset_labels iff o_span in BERT_ev
				o_span.label = offset_labels[k]

def add_ner_output(docs, ner_fname, verbose = True):
	if not docs[0].has_sf_lf_map():
		print('Warning: apply replace_acronyms first or the offsets may be wrong!')
	doc_lookup = { d.id: d for d in docs }
	rows = [json.loads(l.strip()) for l in open(ner_fname).readlines()]
	for row in rows:
		if row['pmid'] not in doc_lookup:
			continue
		doc = doc_lookup[row['pmid']]
		e_label_ranges = utils.condense_labels(row['pred_labels'], '0')
		for i, f, l in e_label_ranges:
			if l not in NER_LABEL_MAP:
				if verbose: print('skipping ner data with unknown label: {}'.format(l))
				continue
			text_i = row['offsets'][i][0]
			text_f = row['offsets'][f-1][1]
			span = classes.Span(text_i, text_f, doc.text[text_i:text_f])
			doc.labels['NER_'+NER_LABEL_MAP[l]].append(span)

def get_doc_spans(doc, label_prefix):
	valid_labels = [l for l in doc.labels if l.startswith(label_prefix)]
	if not valid_labels:
		print('Warning! Unable to find valid labels for {}'.format(label_prefix))
	valid_spans = [s for l in valid_labels for s in doc.labels[l]]
	return valid_spans

def print_ner_token_labels(doc):
	pred_labels = get_ner_labels(doc, 'NER')
	true_labels = get_ner_labels(doc, 'GOLD')
	for tok, t, p in zip(doc.tokens, true_labels, pred_labels):
		print('{} {} {} {}'.format(t, p, tok.label, tok.text))

def print_ner_data(fname, pmid):
	rows = [json.loads(l.strip()) for l in open(fname).readlines()]
	rows = [r for r in rows if r['pmid'] == pmid]
	for r in rows:
		for t, s, l in zip(*[r['tokens'], r['offsets'], r['pred_labels']]):
			print('{}\t{}\t{}-{}'.format(l if l != '0' else '', t, s[0], s[1]))

def print_i_clusters(doc, label_prefix = 'NER_i', thresh = 5):
	spans = get_doc_spans(doc, label_prefix)
	cluster_spans = defaultdict(list)
	if len(spans) > 2:
		embs = encode([s.text for s in spans])
		model = agg_cluster(n_clusters=None, \
				affinity='euclidean', linkage='ward', distance_threshold=thresh)
		for cluster, span in zip(model.fit_predict(embs), spans):
			cluster_spans[cluster].append(span)
		for c in cluster_spans.items():
			print(c)

"""

Doc => Entity list extraction

"""

# BERT-encode NER spans, and cluster them in to distinct entities
def get_cluster_entities(doc, label_prefix = 'NER', thresh = 5, assign_mentions = False):
	entities = []
	for e in 'io':
		spans = get_doc_spans(doc, label_prefix + '_' + e)
		cluster_spans = defaultdict(list)
		# back off in the degenerate case of only one span
		if len(spans) == 1:
			cluster_spans[0].append(spans[0])
		else:
			embs = encode([s.text for s in spans])
			model = agg_cluster(n_clusters=None, \
					affinity='euclidean', linkage='ward', distance_threshold=thresh)
			for cluster, span in zip(model.fit_predict(embs), spans):
				cluster_spans[cluster].append(span)
		# take the first span in each cluster as the Entity name
		for cluster, spans in cluster_spans.items():
			entity = classes.Entity(spans[0], e, cluster)
			if assign_mentions:
				entity.mentions = spans
			entities.append(entity)
	return entities

# Extract each ev-inf I/C/O span as a distinct entity
def get_frame_entities(doc):
	# collapse Entities with identical text
	entities = {}
	for frame in doc.frames:
		if frame.i.text not in entities: entities[frame.i.text] = classes.Entity(frame.i, 'i')
		if frame.c.text not in entities: entities[frame.c.text] = classes.Entity(frame.c, 'i')
		if frame.o.text not in entities: entities[frame.o.text] = classes.Entity(frame.o, 'o')
	return list(entities.values())

# Use each gold-standard coref cluster as a distinct entity
def get_gold_entities(doc, assign_mentions = False):
	# collapse Entities with identical coref groups
	entities = []
	for l in doc.labels:
		if l.startswith('GOLD_'):
			_, e, g_name = l.split('_')
			entity = classes.Entity(classes.Span(-1, -1, g_name), e)
			if assign_mentions:
				entity.mentions = doc.labels[l]
			entities.append(entity)
	return entities

"""

Entity, Doc => Mention assignment

"""

def assign_exact_mention(entities, doc):
	for e in entities:
		e.mentions = []
		for m in re.finditer(e.text, doc.text):
			e.mentions.append(classes.Span(m.start(), m.end(), e.text))

def assign_best_mention(entities, doc):
	mentions = { \
			'i': [m for m in get_doc_spans(doc, 'NER_i') if m.text and not m.text.isspace()],
			'o': [m for m in get_doc_spans(doc, 'NER_o') if m.text and not m.text.isspace()]
	}
	embs = { \
			'i': encode([m.text for m in mentions['i']]),
			'o': encode([m.text for m in mentions['o']])
	}
	for e in entities:
		e_emb = encode([e.text])[0]
		dists = [cos_dist(e_emb, m_emb) for m_emb in embs[e.type]]
		sorted_dists = sorted(zip(dists, mentions[e.type]), key = itemgetter(0))
		if sorted_dists[0][0] >= 0.15:
			continue
		e.mentions = [sorted_dists[0][1]]

# Assign each NER span to the closest entity in embedding space
def assign_bert_mentions(entities, doc, label_prefix = 'NER', \
		max_dist = 0.10, add_unlinked_entities = True):
	for t in 'io':
		valid_entities = [e for e in entities if e.type == t]
		valid_mentions = get_doc_spans(doc, label_prefix + '_'+ t)
		valid_mentions = [m for m in valid_mentions if m.text and not m.text.isspace()]
		if not valid_mentions:
			print('Warning! No valid mentions for {} ({})'.format(doc.id, t))
			continue
		try:
			entity_embs = list(encode([e.text for e in valid_entities]))
			mention_embs = encode([m.text for m in valid_mentions])
		except ValueError:
			print(doc.id)
			print(t)
			print(valid_mentions)
			raise
		for m, m_emb in zip(valid_mentions, mention_embs):
			dists = [cos_dist(m_emb, e_emb) for e_emb in entity_embs]
			# explicitly sort on first element - builtin comparator breaks when dists are tied!
			sorted_dists = sorted(zip(dists, valid_entities), key = itemgetter(0))
			# require a minimum similarity between mention and entity
			if sorted_dists[0][0] <= max_dist:
				sorted_dists[0][1].mentions.append(m)
			else:
				if add_unlinked_entities:
					# ooohhh sheeeeeit create a new entity
					unlinked_e = classes.Entity(m, t)
					unlinked_e.mentions.append(m)
					entities.append(unlinked_e)
					valid_entities.append(unlinked_e)
					entity_embs.append(encode([unlinked_e.text])[0])

"""

Entity, Doc => Entity name assignment

"""

# Use the text span used to instantiate the Entity
def assign_text_names(entities, doc):
	for e in entities:
		e.name = e.text

def assign_metamap_names(entites, doc):
	raise NotImplementedError

"""

Entities, Doc => Entity relations

"""

def get_frame_rel(label, invert = False):
	if label == 0:
		return 'SAME'
	if invert:
		label = -1*label
	if label == 1:
		return 'INCR'
	if label == -1:
		return 'DECR'
	return label

def span_text(span):
	return span.text

def span_gold_label(span):
	return span.label[2:] # trim off the "i_" prefix from the group name

def get_frame_relations(entities, doc, span_name_fn = span_text, add_nulls = False):
	eps = {}
	es = { e.name: e for e in entities }

	def add_rel(s1, s2, rel, overwrite = True):
		nonlocal eps
		if s1 in es and s2 in es:
			if overwrite or (s1, s2) not in eps:
				eps[(s1, s2)] = rel

	for frame in doc.frames:
		i_name = span_name_fn(frame.i)
		o_name = span_name_fn(frame.o)
		add_rel(i_name, o_name, get_frame_rel(frame.label))

	i_names = [e.name for e in entities if e.type == 'i']
	o_names = [e.name for e in entities if e.type == 'o']
	if add_nulls:
		for i, o in itertools.product(i_names, o_names):
			add_rel(i, o, 'NULL', overwrite = False)

	for (i, o), r in eps.items():
		es[i].relations.append((o, r))
		es[o].relations.append((i, r))

	return eps

def get_dummy_relations(entities, doc):
	eps = {}
	i_entities = [e for e in entities if e.type == 'i']
	o_entities = [e for e in entities if e.type == 'o']
	for i, o in itertools.product(i_entities, o_entities):
		eps[(i.text, o.text)] = 'NULL'
	return eps

def no_op(entities, doc):
	return


"""

Doc => Entities, Relations processing

"""

def extract_doc_info(doc, entity_fn, mention_fn, naming_fn, relation_fn, drop_mentionless = True):
	try:
		entities = []
		eps = {}
		entities = entity_fn(doc)
		mention_fn(entities, doc)
		if drop_mentionless:
			entities = [e for e in entities if e.mentions]
		naming_fn(entities, doc)
		eps = relation_fn(entities, doc)
	except Exception as e:
		print('ERROR! Caught exception extracting info from {}. Returning empty data'.format(doc.id))
		traceback.print_exc()
		eps = eps or {}
		entities = entities or []
	doc.entities = entities
	doc.relations = eps
	return entities, eps

def extract_dygie_info(doc):
	return extract_doc_info(doc, get_frame_entities, \
			assign_best_mention, assign_text_names, get_frame_relations)

def extract_distant_info(doc):
	return extract_doc_info(doc, get_frame_entities, \
			assign_bert_mentions, assign_text_names, get_frame_relations)

def extract_gold_info(doc):
	return extract_doc_info(doc, partial(get_gold_entities, assign_mentions = True), \
			no_op, assign_text_names, partial(get_frame_relations, span_name_fn = span_gold_label))

def extract_unsupervised_info(doc):
	return extract_doc_info(doc, \
		partial(get_cluster_entities, assign_mentions = True), \
				no_op, assign_text_names, get_dummy_relations)

def clear_doc_entities(doc):
	del doc.entities
	del doc.relations
	
def set_phase1_gold(docs):
	for d in docs:
		d.labels['BERT_ev'] = [s for s in d.sents if any([utils.s_overlap(s, f.ev) for f in d.frames])]
		d.labels['NER_i'] = get_doc_spans(d, 'GOLD_i')
		d.labels['NER_o'] = get_doc_spans(d, 'GOLD_o')

"""

Evaluation stuff
TODO: move to different file

"""

def get_span_entities(doc, span, e_type):
	return { e for e in doc.entities if e.type == e_type and utils.s_overlaps(span, e.mentions) }

def get_coref_scores(entities_1, entities_2, avg_metrics = True):
	mentions_1 = [{(s.i, s.f, s.text) for s in e.mentions} for e in entities_1 if e.mentions]
	mentions_2 = [{(s.i, s.f, s.text) for s in e.mentions} for e in entities_2 if e.mentions]
	scores = [f(mentions_1, mentions_2) for f in [b_cubed, muc, ceaf_e]]
	if avg_metrics:
		return [np.mean(xs) for xs in zip(*scores)]
	else:
		return scores

FUNCTION_POS = ['-LRB-', '-RRB-', '.', ',', 'CC', 'DT', 'IN']

def eval_gold_linking(docs):
	correct = { e: 0 for e in 'ico' }
	total = 0
	for d in docs:
		for ev in d.labels['BERT_ev']:
			pred_i = { e.name for e in get_span_entities(d, ev.pred_i, 'i') }
			pred_c = { e.name for e in get_span_entities(d, ev.pred_c, 'i') }
			assert len(pred_i) == len(pred_c) == 1
			pred_o = set()
			for o in ev.pred_os:
				pred_o.update({ e.name for e in get_span_entities(d, o, 'o') })

			for f in d.frames:
				if utils.s_overlap(ev, f.ev):
					total += 1
					if f.i.label.split('_')[1] in pred_i: correct['i'] += 1 
					if f.c.label.split('_')[1] in pred_c: correct['c'] += 1 
					if f.o.label.split('_')[1] in pred_o: correct['o'] += 1 
	for e in 'ico':
		print(e, correct[e]/total)


def eval_linking_entities(docs):
	tp = { e: 0 for e in 'ico' }
	fp = { e: 0 for e in 'ico' }
	fn = { e: 0 for e in 'ico' }
	for d in docs:
		for ev in d.labels['BERT_ev']:
			print(ev, d.id)
			pred = {}
			pred['i'] = { e.name for e in get_span_entities(d, ev.pred_i, 'i') }
			pred['c'] = { e.name for e in get_span_entities(d, ev.pred_c, 'i') }
			pred['o'] = set()
			for pred_o in ev.pred_os:
				pred['o'].update({ e.name for e in get_span_entities(d, pred_o, 'o') })

			true = { e: set() for e in 'ico' }
			true_frames = [f for f in d.frames if utils.s_overlap(ev, f.ev)]
			for f in true_frames:
				true['i'].add(f.i.label.split('_')[1])
				true['c'].add(f.c.label.split('_')[1])
				true['o'].add(f.o.label.split('_')[1])

			for e in 'ico':
				tp[e] += len(pred[e] & true[e])
				fp[e] += len(pred[e] - true[e])
				fn[e] += len(true[e] - pred[e])

	for e in 'ico':
		p = tp[e] / (tp[e] + fp[e])
		r = tp[e] / (tp[e] + fn[e])
		f1 = 2*(p * r)/(p + r)
		print('{}\t{:.2f}\t{:.2f}\t{:.2f}'.format(e, p, r, f1))


def get_ner_labels(doc, prefix = 'NER', neg_label = '0', pos_filter = None):
	final_labels = []
	for token, token_labels in zip(doc.tokens, doc.get_token_labels(multi_label = True)):
		valid_labels = [l for l in token_labels if l.startswith(prefix)]
		if len(valid_labels) == 0:
			label = neg_label
		elif pos_filter and token.label in pos_filter:
			label = neg_label
		else:
			label = valid_labels[0].split('_')[1] # labels are {prefix}_{type}_{extra}
		final_labels.append(label)
	return final_labels

def ner_token_score(docs, true_prefix = 'GOLD', pred_prefix = 'NER'):
	true = []
	pred = []
	for d in docs:
		true += get_ner_labels(d, true_prefix)
		pred += get_ner_labels(d, pred_prefix)
	labels = ['i', 'o']
	print('{}\t{}\t{}\t{}'.format('', 'p', 'r', 'f1'))
	for l in labels:
		p, r, f1, _ = precision_recall_fscore_support(true, pred, labels = [l], average = 'micro')
		print('{}\t{:.2f}\t{:.2f}\t{:.2f}'.format(l, p, r, f1))
	p, r, f1, _ = precision_recall_fscore_support(true, pred, labels = labels, average = 'micro')
	return { 'precision': p, 'recall': r, 'f1': f1 }

def ner_span_score(docs, true_prefix, pred_prefix):
	tp = 0
	fp = 0
	fn = 0
	for doc in docs:
		pred_spans = get_doc_spans(doc, pred_prefix) 
		true_spans = get_doc_spans(doc, true_prefix) 
		for pred in pred_spans:
			if utils.s_overlaps(pred, true_spans):
				tp += 1
			else:
				fp += 1
		for true in true_spans:
			if utils.s_overlaps(true, pred_spans):
				pass # already counted the TP
			else:
				fn += 1
	p = tp / (tp + fp)
	r = tp / (tp + fn)
	f1 = 2*(p * r)/(p + r)
	return p, r, f1

def ner_entity_score(docs, true_prefix = 'GOLD', pred_prefix = 'NER', e_types = ['i', 'o']):
	tp = 0
	fn = 0
	fp = 0
	for doc in docs:
		pred_spans = get_doc_spans(doc, pred_prefix)
		true_spans = get_doc_spans(doc, true_prefix)
		for e in doc.entities:
			if e.type in e_types:
				found = False
				for m in e.mentions:
					if utils.s_overlaps(m, pred_spans):
						found = True
						break
				# count any entity as a true positive if any of its mentions are tagged
				if found:
					tp += 1
				# and a false negative otherwise
				else:
					fn += 1
		# for false positives, we're trying to count how many extraneous "entities" there are
		# we need some notion of which pred spans are the same entity - lets be pessimistic
		# and say they're different entities unless they are exactly the same (overcount FPs)
		e_func = lambda s: s.text
		# to count false positives, take every pred span
		fp_spans = { e_func(s): 1 for s in pred_spans }
		for s in pred_spans:
			# and don't count ones that overlap a true mention
			if utils.s_overlaps(s, true_spans):
				fp_spans[e_func(s)] = 0
		fp += sum(fp_spans.values())
	p = tp / (tp + fp)
	r = tp / (tp + fn)
	f1 = 2*(p * r)/(p + r)
	return p, r, f1

def frame_ev(doc, frame):
	return [frame.ev]

def full_context(doc, frame):
	group_idx = doc.text.lower().find('group')
	sents = [s for idx, s in enumerate(doc.sents) if \
				utils.s_overlap(s, frame.ev) or \
				s.i <= group_idx <= s.f or \
				2 <= idx <= 4]
	return sents

def doc_intro(doc, frame, n = 5):
	return doc.sents[:n]

def doc_token_perc(doc, frame, p = 0.25):
	return doc.tokens[:int(len(doc.tokens)*p)]

def full_doc(doc, frame):
	return doc.sents

def ner_frame_score(docs, span_fn, pred_prefix = 'NER', e_type = 'i'):
	tp = 0
	fn = 0
	for doc in docs:
		true_labels = doc.get_char_labels('GOLD_{}'.format(e_type))
		pred_labels = doc.get_char_labels('{}_{}'.format(pred_prefix, e_type))
		char_labels = [true if pred else set() for true, pred in zip(true_labels, pred_labels)]
		for frame in doc.frames:
			if frame.label == 0:
				continue
			valid_spans = span_fn(doc, frame)
			valid_labels = set()
			for s in valid_spans:
				valid_labels.update(utils.unioned(char_labels[s.i:s.f]))
			if 'GOLD_{}'.format(getattr(frame, e_type).label) in valid_labels:
				tp += 1
			else:
				print()
				print(frame.i, doc.id)
				for s in valid_spans:
					print('\t', s)
				fn += 1
	r = tp / (tp + fn)
	print('I: {} ({}, {})'.format(r, tp, fn))
	return r

def eval_ner(docs, scorer,
			true_processor = partial(get_ner_labels, prefix = 'GOLD'),
			pred_processor = partial(get_ner_labels, prefix = 'NER')):
	true = []
	pred = []
	for d in docs:
		true += true_processor(d)
		pred += pred_processor(d)
	scores = scorer(true, pred)
	return scores

def eval_ic_overlap(docs):
	n_i = 0
	n_ic = 0
	label_counts = { x:0 for x in [-1,0,1] }
	for doc in docs:
		i = set([f.i.text for f in doc.frames])
		c = set([f.c.text for f in doc.frames])
		ic = i & c
		n_i += len(i)
		n_ic += len(ic)
		for t in ic:
			for f in doc.frames:
				if f.c.text == t or f.i.text == t:
					label_counts[f.label] += 1
					print(f.label, '|', f.i.text, '|', f.c.text)
	print(label_counts)
	print(n_i, n_ic, n_ic/n_i)

def eval_o_consistency(docs):
	n = 0
	n_mult = 0
	for doc in docs:
		for o in set([f.o.text for f in doc.frames]):
			n += 1
			labels = [f.label for f in doc.frames if f.o.text == o]
			if len(set(labels)) > 1:
				n_mult += 1
				print(set(labels))
	print(n, n_mult)

def eval_ev_i_consistency(docs):
	n = 0
	n_mult = 0
	for doc in docs:
		ev_i_map = defaultdict(set)
		for frame in doc.frames:
			ev_i_map[frame.ev.text].add(frame.i.text)
		n += len(ev_i_map)
		for e, i in ev_i_map.items():
			if len(i) > 1:
				n_mult += 1
				print(e)
				print(i)
				print()
	print(n, n_mult)

def parse_iain_inactive_data(fname):
	raw_entries = open(fname).read().split('- ')[1:]
	rows = []
	for e in raw_entries:
		lines = [l.strip() for l in e.split('\n')[:-1]]
		kv_pairs = [l.split(': ') for l in lines]
		rows.append({ kv[0]: kv[1] for kv  in kv_pairs if len(kv) == 2 })
	i = []
	c = []

	def parse_graph_label(s):
		for prefix in ['Favours ', 'favours ']:
			if s.startswith(prefix):
				s = s[len(prefix):]
		return s

	for row in rows:
		if 'control_arm' not in row:
			continue

		if row['control_arm'] == '0':
			pass
		elif row['control_arm'] == '1':
			c.append(parse_graph_label(row['graph_label_1']))
			i.append(parse_graph_label(row['graph_label_2']))
		elif row['control_arm'] == '2':
			c.append(parse_graph_label(row['graph_label_2']))
			i.append(parse_graph_label(row['graph_label_1']))
	return i, c

def print_doc_labels(doc):
	print(doc.id)
	for l, spans in doc.labels.items():
		print(l)
		for s in spans:
			print('\t{}'.format(s.text))

def print_entities(entities):
	for e in entities:
		print(e.text)
		for m in e.mentions:
			print('\t{}'.format(m.text))
		print()

def print_first_instance(doc, s):
	print(utils.s_overlaps(s, doc.sents))

def print_frames(doc):
	for idx, f in enumerate(doc.frames):
		print('Frame {} [{}]'.format(idx, f.label))
		print('\tI: {}'.format(f.i.text))
		print('\tC: {}'.format(f.c.text))
		print('\tO: {}'.format(f.o.text))
		print('\tEV: {}'.format(f.ev.text))

def export_json_docs(docs):
	rows = []
	def sencode(s):
		return { 'text': s.text, 'i': s.i, 'f': s.f, \
			'concepts': [c.Concept_Name for c in s.concepts] if s.concepts else [] }
	for d in docs:
		r = {}
		r['pmid'] = d.id
		r['text'] = d.text
		r['frames'] = []
		for f in d.frames:
			r['frames'].append({\
				'i': sencode(f.i),
				'c': sencode(f.c),
				'o': sencode(f.o),
				'ev': [sencode(e) for e in f.evs]})
		r['ner'] = {\
			'i': [sencode(s) for s in d.ner['i']],
			'o': [sencode(s) for s in d.ner['o']],
		}
		rows.append(r)
	return rows

def chars_to_tokens(chars, tokens):
	t_i = -1
	t_f = -1
	for idx, t in enumerate(tokens):
		if t.i <= chars.i <= t.f:
			t_i = idx
		if t.i <= chars.f <= t.f:
			t_f = idx
	return [t_i, t_f]

def export_json(docs, super_fn = extract_distant_info):
	rows = []
	for d in docs:
		if super_fn:
			es, eps = super_fn(d)
		else:
			es, eps = d.entities, d.relations
		r = {}
		r['text'] = d.text
		r['id'] = d.id
		r['entities'] = []
		for e in es:
			r['entities'].append({
				'text': e.text,
				'mentions': list(set([(s.i, s.f, s.text) for s in e.mentions])),
				'class': e.type
			})
		r['relations'] = []
		for ep, rel in eps.items():
			r['relations'].append({
				'e1': ep[0],
				'e2': ep[1],
				'class': rel
			})
		rows.append(r)
	return rows
