import utils
import classes
from ner_scores import compute_metrics as ner_score
from coref_scores import b_cubed, muc, ceaf_e
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from sklearn.cluster import AgglomerativeClustering as agg_cluster
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import cosine as cos_dist
from stanfordnlp import Pipeline
import sys
import random
import json
import re
import pickle as pkl
from collections import defaultdict, Counter
from operator import itemgetter
from functools import partial
import traceback

from bert_serving.client import BertClient
BC = None
SF = None


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
NER_LABEL_MAP = {
    'I_int': 'i',
    'I_out': 'o',
    'i': 'i',
    'o': 'o',
}


def add_ner_output(docs, ner_fname):
    if not hasattr(docs[0], 'sf_lf_map'):
        # fml
        print('ERROR: apply replace_acronyms first or the offsets are wrong!')
        return
    doc_lookup = {d.id: d for d in docs}
    rows = [json.loads(l.strip()) for l in open(ner_fname).readlines()]
    for row in rows:
        if row['pmid'] not in doc_lookup:
            continue
        doc = doc_lookup[row['pmid']]
        e_label_ranges = utils.condense_labels(row['pred_labels'], 'O')
        for i, f, l in e_label_ranges:
            if l not in NER_LABEL_MAP:
                print('skipping ner data with unknown label: {}'.format(l))
                continue
            text_i = row['offsets'][i][0]
            text_f = row['offsets'][f-1][1]
            span = classes.Span(text_i, text_f, doc.text[text_i:text_f])
            doc.labels['NER_'+NER_LABEL_MAP[l]].append(span)


def get_doc_spans(doc, label_prefix, e):
    t_label_prefix = label_prefix + '_' + e
    valid_labels = [l for l in doc.labels if l.startswith(t_label_prefix)]
    if not valid_labels:
        print('Warning! Unable to find valid labels for {}+{}'.format(label_prefix, e))
    valid_spans = [s for l in valid_labels for s in doc.labels[l]]
    return valid_spans


"""

Doc => Entity list extraction

"""

# BERT-encode NER spans, and cluster them in to distinct entities


def get_cluster_entities(doc, label_prefix='NER', thresh=5, assign_mentions=False):
    entities = []
    for e in 'io':
        spans = get_doc_spans(doc, label_prefix, e)
        cluster_spans = defaultdict(list)
        # back off in the degenerate case of only one span
        if len(spans) == 1:
            cluster_spans[0].append(spans[0])
        else:
            embs = encode([s.text for s in spans])
            model = agg_cluster(n_clusters=None,
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
        if frame.i.text not in entities:
            entities[frame.i.text] = classes.Entity(frame.i, 'i')
        if frame.c.text not in entities:
            entities[frame.c.text] = classes.Entity(frame.c, 'i')
        if frame.o.text not in entities:
            entities[frame.o.text] = classes.Entity(frame.o, 'o')
    return list(entities.values())

# Use each gold-standard coref cluster as a distinct entity


def get_gold_entities(doc, assign_mentions=False):
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

# Assign each NER span to the closest entity in embedding space
def assign_bert_mentions(entities, doc, label_prefix = 'NER', \
		max_dist = 0.10, add_unlinked_entities = False):
	for t in 'io':
		valid_entities = [e for e in entities if e.type == t]
		valid_mentions = get_doc_spans(doc, label_prefix + '_'+ t)
		valid_mentions = [m for m in valid_mentions if not m.text.isspace()]
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


def get_frame_rel(label, invert=False):
    if label == 0:
        return 'SAME'
    if invert:
        label = -1*label
    if label == 1:
        return 'INCR'
    if label == -1:
        return 'DECR'
    return label

import itertools
def get_frame_relations(entities, doc):
    eps = {}
    es = {e.text: e for e in entities}

    def add_rel(s1, s2, rel, overwrite=True):
        nonlocal eps
        if s1 in es and s2 in es:
            if (s1, s2) not in eps or overwrite:
                eps[(s1, s2)] = rel

    for frame in doc.frames:
        add_rel(frame.i.text, frame.o.text, get_frame_rel(frame.label))
        add_rel(frame.c.text, frame.o.text,
                get_frame_rel(frame.label, invert=True))
    i_entities = [e for e in entities if e.type == 'i']
    o_entities = [e for e in entities if e.type == 'o']
    for i, o in itertools.product(i_entities, o_entities):
        add_rel(i.text, o.text, 'NULL', overwrite=False)
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


def extract_doc_info(doc, entity_fn, mention_fn, naming_fn, relation_fn):
    try:
        entities = entity_fn(doc)
        mention_fn(entities, doc)
        naming_fn(entities, doc)
        eps = relation_fn(entities, doc)
        return entities, eps
    except:
        print('ERROR! Caught exception extracting info from {}. Returning empty data'.format(doc.id))
        return [], {}


def extract_distant_info(doc):
	return extract_doc_info(doc, get_frame_entities, \
			partial(assign_bert_mentions, add_unlinked_entities = True), \
			assign_text_names, get_frame_relations)


def extract_gold_info(doc):
    return extract_doc_info(doc, partial(get_gold_entites, assign_mentions=True),
                            no_op, assign_text_names, get_frame_relations)


def extract_unsupervised_info(doc):
    return extract_doc_info(doc,
                            partial(get_cluster_entities,
                                    assign_mentions=True),
                            no_op, assign_text_names, get_dummy_relations)


"""

Evaluation stuff
TODO: move to different file

"""


def get_coref_scores(entities_1, entities_2):
    mentions_1 = [{(s.i, s.f, s.text) for s in e.mentions}
                  for e in entities_1 if e.mentions]
    mentions_2 = [{(s.i, s.f, s.text) for s in e.mentions}
                  for e in entities_2 if e.mentions]
    scores = [f(mentions_1, mentions_2) for f in [b_cubed, muc, ceaf_e]]
    return [np.mean(xs) for xs in zip(*scores)]


def fuzzy_match_spans(true_spans, pred_spans):
    span_map = {}


def eval_coref(docs, true_processor, pred_processor, matching='exact'):
    scores = [get_coref_scores(true_processor(
        d), pred_processor(d)) for d in docs]
    return [np.mean(xs) for xs in zip(*scores)]


def get_ner_labels(doc, prefix='NER', neg_label='0'):
    final_labels = []
    for token_labels in doc.get_token_labels():
        valid_labels = [l for l in token_labels if l.startswith(prefix)]
        if not valid_labels:
            label = neg_label
        else:
            assert len(valid_labels) == 1
            # labels are {prefix}_{type}_{extra}
            label = valid_labels[0].split('_')[1]
        final_labels.append(label)
    return final_labels


def ner_token_score(true, pred):
    p, r, f1, _ = precision_recall_fscore_support(
        true, pred, labels=['i', 'o'])
    return p, r, f1


def ner_span_score(true, pred, overlap='any', type_match=True):
    true_spans = {s: idx for idx, s in enumerate(utils.condense_labels(true))}
    pred_spans = {s: idx for idx, s in enumerate(utils.condense_labels(pred))}

    # TODO: optimize? 2*n^2
    true_span_labels = []
    for ti, tf, tl in true_spans:
        found = False
        for pi, pf, pl in pred_spans:
            if not type_match or tl == pl:
                if utils.overlap(ti, tf, pi, pf):
                    found = True
                    break
        true_span_labels.append(found)

    pred_span_labels = []
    for pi, pf, pl in pred_spans:
        found = False
        for ti, tf, tl in true_spans:
            if not type_match or tl == pl:
                if utils.overlap(ti, tf, pi, pf):
                    found = True
                    break
        pred_span_labels.append(found)

    tp = sum(pred_span_labels)
    fp = len(pred_span_labels) - tp
    fn = len(true_span_labels) - sum(true_span_labels)

    p = tp / (tp + fp)
    r = tp / (tp + fn)
    f1 = 2*(p * r)/(p + r)
    return p, r, f1


def eval_ner(docs,
             true_processor=partial(get_ner_labels, prefix='GOLD'),
             pred_processor=partial(get_ner_labels, prefix='NER'),
             scorer=ner_token_score):
    scores = [scorer(true_processor(d), pred_processor(d)) for d in docs]
    return [np.mean(xs) for xs in zip(*scores)]


def print_doc_labels(doc):
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


def print_frames(doc):
    for idx, f in enumerate(doc.frames):
        print('Frame {} [{}]'.format(idx, f.label))
        print('\tI: {}'.format(f.i.text))
        print('\tC: {}'.format(f.c.text))
        print('\tO: {}'.format(f.o.text))


def export_json(docs):
    rows = []

    def sencode(s):
        return {'text': s.text, 'i': s.i, 'f': s.f,
                'concepts': [c.Concept_Name for c in s.concepts] if s.concepts else []}
    for d in docs:
        r = {}
        r['pmid'] = d.id
        r['text'] = d.text
        r['frames'] = []
        for f in d.frames:
            r['frames'].append({
                'i': sencode(f.i),
                'c': sencode(f.c),
                'o': sencode(f.o),
                'ev': [sencode(e) for e in f.evs]})
        r['ner'] = {
            'i': [sencode(s) for s in d.ner['i']],
            'o': [sencode(s) for s in d.ner['o']],
        }
        rows.append(r)
    return rows
