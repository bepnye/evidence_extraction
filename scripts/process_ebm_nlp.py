import sys
import os
import json
from glob import glob
from shutil import copyfile

import pandas as pd

sys.path.append('..')
import config
import utils
import classes

GROUPS = ['test', 'train', 'dev']
def read_docs(phase = 'starting_spans'):
	pmid_groups = {}
	for g in GROUPS:
		pmids = utils.readlines(os.path.join(config.EBM_NLP_DIR, 'pmids_{}.txt'.format(g)))
		for pmid in pmids:
			pmid_groups[pmid] = g

	def get_e_fname(pmid, e):
		if pmid_groups[pmid] == 'test':
			subdir = os.path.join('test', 'gold')
		else:
			subdir = 'train'
		f = '{}.AGGREGATED.ann'.format(pmid)
		return os.path.join(config.EBM_NLP_DIR, 'annotations', 'aggregated', phase, e, subdir, f)

	docs = []
	for pmid, group in pmid_groups.items():
		tokens = utils.readlines(os.path.join(config.EBM_NLP_DIR, 'documents', '{}.tokens'.format(pmid)))
		text, token_offsets = utils.join_tokens(tokens)
		doc = classes.Doc(pmid, text)
		doc.group = group
		for e in ['participants', 'interventions', 'outcomes']:
			label_name = 'GOLD_{}'.format(e[0])
			labels = [int(l) for l in utils.readlines(get_e_fname(pmid, e))]
			for token_i, token_f, l in utils.condense_labels(labels):
				char_i = token_offsets[token_i][0]
				char_f = token_offsets[token_f-1][1]
				doc.labels[label_name].append(classes.Span(char_i, char_f, text[char_i:char_f]))
		docs.append(doc)
	return docs
