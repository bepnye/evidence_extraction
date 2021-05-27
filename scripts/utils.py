import os
from random import sample
from collections import namedtuple
from itertools import groupby
from nltk import tokenize as tokenizer
import numpy as np

frame_cols = 'i c o label evidence ev_i ev_f sent_indices'.split()
Frame = namedtuple('Frame', frame_cols)
Sent = namedtuple('Sent', 'i f s')

DATA_DIR = '../data/'
NER_INPUT_FIELDS = 'token pmid token_idx sent_idx dummy'
SENT_INPUT_FIELDS = 'dummy pmid sent_idx sent'

LABEL_TO_ID = { \
				'significantly decreased':	 '0',
				'no significant difference': '1',
				'significantly increased':	 '2', }

def init_data_dir():
	for d in ['documents',
						'documents/frames',
						'documents/sents',
						'documents/title',
						'documents/tokens',
						'documents/txt',
						'id_splits']:
		os.system('mkdir -p {}/{}'.format(DATA_DIR, d))

def drop_none(l):
	return list(filter(None, l))

def safe_div(x, y):
	return x/y if y != 0 else 0

def jaccard(a, b):
	sa = set(a)
	return len(sa.intersection(b))/len(sa.union(b))

def clean_str(s):
	return str(s).replace('\t', ' ').replace('\n', ' ').replace('\r', ' ')

def unioned(l):
	return set.union(*l)

def shuffled(l):
	return sample(l, len(l))

def split_at(l, idx):
	return l[:idx], l[idx:]

def write_line(fd, s, is_first):
	if not is_first:
		fd.write('\n')
	fd.write(s)

def joinstr(seq, sep = '\t'):
	return sep.join([str(s) for s in seq])

def readlines(fname):
	return open(fname).read().strip('\n').split('\n')

def read_tsv(fname):
	return [l.split('\t') for l in open(fname).read().strip('\n').split('\n')]

def read_frames(fname):
	frames = []
	for l in readlines(fname):
		try:
			f = Frame(*l.split('\t'))
			frames.append(f)
		except TypeError:
			print(l)
			raise
	return frames

def mode(l):
	return max(l, key = l.count)

def argmax(l):
	return l.index(max(l))

# set a threshold for the positive class in binary classification
def prob_thresh(l, thresh = 0.5):
	assert len(l) == 2
	return 1 if l[1] >= thresh else 0

def group_ids(dataset, group):
	return readlines('{}/id_splits/{}/{}.txt'.format(DATA_DIR, dataset, group))

def norm_d(d):
	N = sum(d.values())
	return { k: v/N for k,v in d.items() }

def overlap(a1, a2, b1, b2):
	return int(a1) <= int(b2-1) and int(b1) <= int(a2-1)

def s_overlap(s1, s2):
	return overlap(s1.i, s1.f, s2.i, s2.f)

def s_overlaps(target, spans):
	return [s for s in spans if s_overlap(target, s)]

def contained(x1, x2, y1, y2):
	return x1 >= y1 and x2 <= y2

def frame_overlap(s1, s2):
	x1, x2 = s1.ev_i, s1.ev_f
	y1, y2 = s2.ev_i, s2.ev_f
	return overlap(x1, x2, y1, y2)

def condense_labels(labels, neg_class = '0'):
	labels = [str(l) for l in labels]
	groups = [(k, sum(1 for _ in g)) for k,g in groupby(labels)]
	spans = []
	i = 0
	for label, length in groups:
		if label != neg_class:
			spans.append((i, i+length, label))
		i += length
	return spans

def get_bi_labels(labels, neg_class = '0'):
	bi_labels = [neg_class for _ in labels]
	for i, f, l in condense_labels(labels, neg_class):
		bi_labels[i] = 'B-{}'.format(l)
		for idx in range(i+1, f):
			bi_labels[idx] = 'I-{}'.format(l)
	return bi_labels

def sent_tokenize(txt):
	return span_tokenize(txt, tokenizer.sent_tokenize)

def word_tokenize(txt):
	return span_tokenize(txt, tokenizer.word_tokenize)

def join_tokens(tokens, sep = ' '):
	text = sep.join(tokens)
	t_lens = [len(t) for t in tokens]
	t_starts = np.cumsum([0] + [len(sep)+l for l in t_lens])
	t_offsets = [(s, s+l) for s, l in zip(t_starts, t_lens)]
	return text, t_offsets
